# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
import textwrap
from typing import List, Dict, Optional, TypedDict, Any

from pyrit.executor.attack.multi_turn import MultiTurnAttackStrategy, MultiTurnAttackContext
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    PromptRequestPiece,
    PromptRequestResponse,
)
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.memory import CentralMemory
from pyrit.score import Scorer


class AgentEntry(TypedDict):
    """Single hop in the multi-agent chain.

    Fields:
      role: Logical role name of the agent (e.g., "strategy_agent").
      agent: Chat-capable target that receives the hop input and returns a reply.
    """
    role: str
    agent: PromptChatTarget


class MultiAgentAttack(MultiTurnAttackStrategy[MultiTurnAttackContext, AttackResult]):
    """Linear multi-agent red-teaming strategy with deterministic traceability.

    Overview:
      - PyRIT is always the **user** in persisted conversations; any agent/target is the
        **assistant**. We encode the true actor identity in `prompt_metadata["mas_role"]`.
      - Each agent has its own conversation_id (CID). We do not mirror messages into the
        session CID beyond an initial seed.
      - Each outer turn produces a new **target conversation**; the list of target CIDs and
        their chronological order is returned via `metadata["target_conversations"]` as
        `{"turn": int, "conversation_id": str}` entries.

    Labels (constants):
      LABEL_SEED: mas_role for the orchestrator seed stored in the session conversation.
      LABEL_TARGET_PROMPT: mas_role for the final prompt sent to the objective target.
      LABEL_TARGET_RESPONSE: mas_role for the objective target's response.

    Invariants:
      - Requests are persisted with role="user"; responses are role="assistant".
      - Actor identity must be read from `prompt_metadata["mas_role"]`, not the chat role.

    Args:
      agent_chain: Ordered list of agents forming the chain (upstream → downstream).
      agent_contexts: Optional per-role text context prepended to the agent input.
      agent_converters: Optional per-role list of PromptConverters. If a role key is present
        with an empty list, **global converters are not applied** (explicit override).
      objective: Natural-language objective (also used as default context.objective fallback).
      system_prompts: Optional per-role system prompts applied to each agent's CID once.
      prompt_converters: Global converters applied when a role has no per-role converters.
      objective_target: The final target to which the last agent's text is sent each turn.
      scorer: Optional scorer; early-stops when `last_score.get_value()` is truthy.
      verbose: Reserved for future logging; currently no side effects.
      max_turns: Upper bound on outer loop iterations.

    Attributes:
      _conv_ids: Map of PromptChatTarget instance → conversation_id.
      _role_to_cid: Map of logical role → conversation_id, for printers/metadata.
      _history: In-memory trace per outer turn: List[List[(role, text)]].
    """

    LABEL_SEED = "orchestrator_seed"
    LABEL_TARGET_PROMPT = "objective_target_prompt"
    LABEL_TARGET_RESPONSE = "objective_target_response"

    def __init__(
        self,
        *,
        agent_chain: List[AgentEntry],
        agent_contexts: Optional[Dict[str, str]] = None,
        agent_converters: Optional[Dict[str, List[PromptConverter]]] = None,
        objective: str,
        system_prompts: Optional[Dict[str, str]] = None,
        prompt_converters: Optional[List[PromptConverter]] = None,
        objective_target: PromptTarget,
        scorer: Optional[Scorer] = None,
        verbose: bool = False,
        max_turns: int = 10,
    ):
        super().__init__(context_type=MultiTurnAttackContext)
        self.agent_chain = agent_chain
        self.agent_contexts = agent_contexts or {}
        self.agent_converters = agent_converters or {}
        self.objective = objective
        self.system_prompts = system_prompts or {}
        self.objective_target = objective_target
        self.scorer = scorer
        self.verbose = verbose
        self.max_turns = max_turns

        self._prompt_converters = prompt_converters or []

        # runtime state
        self._conv_ids: Dict[PromptChatTarget, str] = {}
        self._role_to_cid: Dict[str, str] = {}               # role -> conversation_id
        self._history: List[List[tuple[str, str]]] = []       # per-turn [(role, text)]
        self._inited = False

    def _initialize_agents(self) -> None:
        """Create per-agent conversations and apply system prompts.

        Side effects:
          - Populates `_conv_ids` and `_role_to_cid`.
          - Calls `set_system_prompt(...)` for agents that have a configured system prompt.
        """

        for entry in self.agent_chain:
            role, agent = entry["role"], entry["agent"]
            if agent not in self._conv_ids:
                cid = str(uuid.uuid4())
                self._conv_ids[agent] = cid
                self._role_to_cid[role] = cid
                if sys_prompt := self.system_prompts.get(role):
                    agent.set_system_prompt(system_prompt=sys_prompt, conversation_id=cid)
        self._inited = True

    async def _setup_async(self, *, context: MultiTurnAttackContext) -> None:
        """Prepare runtime state and register conversations for printing.

        Adds all agent conversation IDs into `context.related_conversations` so stock
        printers can render them alongside the session conversation.
        """

        if not self._inited:
            self._initialize_agents()
        self._history = []

        # Ensure agent conversation IDs are registered so printers can render them.
        for _, cid in self._role_to_cid.items():
            context.related_conversations.add(cid)

    async def _perform_async(self, *, context: MultiTurnAttackContext) -> AttackResult:
        """Run the multi-agent loop and collect an `AttackResult`.

        Flow per outer turn:
          1) Build the first agent's input from the incoming `input_text` (seed or previous
             target output). Downstream agents receive the rolling in-turn internal history.
          2) For each agent:
             - Optionally apply per-role/global converters (role-specific overrides first).
             - Persist the request (`role="user"`) and the agent's reply (`role="assistant"`)
               under the agent's own CID, labeling both pieces with `mas_role`.
          3) Send the final agent output to the objective target in a fresh target CID.
             Persist both the prompt (`mas_role=LABEL_TARGET_PROMPT`) and the response
             (`mas_role=LABEL_TARGET_RESPONSE`), and append the CID to
             `metadata["target_conversations"]` with the `turn` index.
          4) If a scorer is configured, score the target response and early-stop on success.
             Otherwise, feed the target output back into the next outer turn.

        Determinism:
          - Target conversations are emitted in the order they occur and tagged with `turn`,
            so renderers should respect list order (do not sort UUIDs).

        Args:
          context: MultiTurnAttackContext carrying the session and optional custom prompt.

        Returns:
          AttackResult with:
            - `outcome`: SUCCESS on early-stop by scorer; FAILURE otherwise;
              UNDETERMINED for early errors (e.g., empty agent outputs, target error).
            - `last_response`: The last target response piece if available.
            - `last_score`: The last computed score if a scorer is configured.
            - `related_conversations`: Union of agent CIDs + target CIDs for printers.
            - `metadata`: Dict with `agent_conversations` and `target_conversations`.

        Raises:
          ValueError: If no objective is provided or `agent_chain` is empty.
        """

        self._validate_context(context=context)

        session_cid = context.session.conversation_id
        input_text = context.custom_prompt or context.objective or self.objective
        achieved = False
        last_score: Any = None

        memory = CentralMemory.get_memory_instance()

        # Seed the orchestrator session exactly once (no mirroring after this).
        try:
            seed_piece = PromptRequestPiece(
                role="user",
                conversation_id=session_cid,
                original_value=input_text,
                converted_value=input_text,
                prompt_metadata={"mas_role": self.LABEL_SEED},
            )
            memory.add_request_response_to_memory(request=PromptRequestResponse([seed_piece]))
        except Exception:
            # Non-fatal; session transcript may appear empty.
            pass

        # Ordered chronology for target conversations
        target_conversations: List[Dict[str, Any]] = []

        for turn_idx in range(self.max_turns):
            turn_trace: List[tuple[str, str]] = []
            internal_history: List[Dict[str, str]] = []  # [{"role": "...", "text": "..."}]
            current_input = input_text

            for agent_index, entry in enumerate(self.agent_chain):
                role = entry["role"]
                agent = entry["agent"]

                # Compose input for this agent
                parts: List[str] = []
                if ctx_text := self.agent_contexts.get(role):
                    parts.append(f"[{role.upper()} CONTEXT]\n{ctx_text}\n")

                # First agent gets the incoming seed; downstream get the internal rolling history
                if internal_history:
                    parts.append("\n".join(f"{h['role']}: {h['text']}" for h in internal_history))
                else:
                    parts.append(current_input)

                full_input = "\n".join(filter(None, parts)).strip()

                # Apply converters (role-specific overrides first, else global)
                for conv in (self.agent_converters.get(role) or self._prompt_converters):
                    out = await conv.convert_async(prompt=full_input)
                    full_input = out.output_text

                # Persist agent request under its own conversation
                req_piece = PromptRequestPiece(
                    role="user",
                    conversation_id=self._conv_ids[agent],
                    original_value=full_input,
                    converted_value=full_input,
                    prompt_metadata={"mas_role": role, "agent_index": agent_index},
                )
                req = PromptRequestResponse(request_pieces=[req_piece])
                memory.add_request_response_to_memory(request=req)

                # Send to agent
                res = await agent.send_prompt_async(prompt_request=req)
                output_text = res.request_pieces[0].converted_value

                # Persist the agent’s reply (assistant) under the agent’s conversation with label
                self._persist_message(
                    agent_role=role,
                    text=output_text,
                    conversation_id=self._conv_ids[agent],
                )

                internal_history.append({"role": role, "text": output_text})
                turn_trace.append((role, output_text))
                current_input = output_text  # seed for next agent

            # If no agent produced output, return UNDETERMINED
            if not internal_history:
                return AttackResult(
                    attack_identifier=self.get_identifier(),
                    conversation_id=session_cid,
                    objective=self.objective,
                    outcome=AttackOutcome.UNDETERMINED,
                    outcome_reason="No agent output produced this turn.",
                    executed_turns=len(self._history),
                    last_response=None,
                    last_score=last_score,
                    related_conversations=context.related_conversations,
                    metadata={
                        "agent_conversations": self._role_to_cid,
                        "target_conversations": target_conversations,
                    },
                )

            # Compose final prompt from the last agent output and send to objective target
            final_prompt = internal_history[-1]["text"]
            tgt_cid = str(uuid.uuid4())
            target_conversations.append({"turn": len(self._history), "conversation_id": tgt_cid})
            context.related_conversations.add(tgt_cid)  # so printers will render it

            # Persist the target prompt with explicit label
            target_req = PromptRequestResponse(request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id=tgt_cid,
                    original_value=final_prompt,
                    converted_value=final_prompt,
                    prompt_metadata={"mas_role": self.LABEL_TARGET_PROMPT, "turn": len(self._history)},
                )
            ])
            memory.add_request_response_to_memory(request=target_req)

            # Call target, guard exceptions so we can return a sensible result
            target_res = None
            try:
                target_res = await self.objective_target.send_prompt_async(prompt_request=target_req)
                # Ensure each response piece is labeled for clear printing
                for p in target_res.request_pieces:
                    meta = dict(getattr(p, "prompt_metadata", {}) or {})
                    meta.setdefault("mas_role", self.LABEL_TARGET_RESPONSE)
                    meta.setdefault("turn", len(self._history))
                    p.prompt_metadata = meta
                memory.add_request_response_to_memory(request=target_res)
            except Exception as e:
                return AttackResult(
                    attack_identifier=self.get_identifier(),
                    conversation_id=session_cid,
                    objective=self.objective,
                    outcome=AttackOutcome.UNDETERMINED,
                    outcome_reason=f"Target error: {e}",
                    executed_turns=len(self._history),
                    last_response=None,
                    last_score=last_score,
                    related_conversations=context.related_conversations,
                    metadata={
                        "agent_conversations": self._role_to_cid,
                        "target_conversations": target_conversations,
                    },
                )

            # Update in-memory trace
            target_output = target_res.request_pieces[0].converted_value
            turn_trace.append(("objective_target", target_output))
            self._history.append(turn_trace)

            # Evaluate success if a scorer is present
            if self.scorer:
                scores = await self.scorer.score_async(request_response=target_res.request_pieces[0])
                last_score = scores[0] if scores else None
                if last_score and last_score.get_value():
                    achieved = True
                    break

            # If not achieved, feed the model’s answer back to the chain
            input_text = target_output

        # Prepare final result
        last_piece = None
        if target_res and getattr(target_res, "request_pieces", None):
            last_piece = target_res.request_pieces[0]

        return AttackResult(
            attack_identifier=self.get_identifier(),
            conversation_id=session_cid,
            objective=self.objective,
            outcome=AttackOutcome.SUCCESS if achieved else AttackOutcome.FAILURE,
            outcome_reason="Score passed threshold" if achieved else "Score threshold not met",
            executed_turns=len(self._history),
            last_response=last_piece,
            last_score=last_score,
            related_conversations=context.related_conversations,
            metadata={
                "agent_conversations": self._role_to_cid,
                "target_conversations": target_conversations,  # chronological with turn indices
            },
        )

    def _persist_message(self, *, agent_role: str, text: str, conversation_id: str) -> None:
        """Persist an agent reply in its own conversation.

        Args:
          agent_role: Logical role (e.g., "strategy_agent") used for `mas_role`.
          text: Agent reply text to store as `assistant`.
          conversation_id: CID under which to persist this message.
        """
        prp = PromptRequestPiece(
            role="assistant",
            original_value=text,
            converted_value=text,
            conversation_id=conversation_id,
            prompt_metadata={"mas_role": agent_role},
        )
        CentralMemory.get_memory_instance().add_request_response_to_memory(
            request=PromptRequestResponse([prp])
        )

    def get_history(self) -> List[List[tuple[str, str]]]:
        """Return the in-memory trace of roles and texts per outer turn.

        Note:
          This is a convenience for debugging/tests. The authoritative transcript
          remains in `CentralMemory`.
        """
        return self._history

    def print_conversation(self, all_turns: bool = True) -> None:
        """Print the in-memory trace (debug utility).

        Args:
          all_turns: If True, prints all collected turns; otherwise prints the last one.

        Warning:
          This prints `_history`, not the persisted `CentralMemory` conversations, and
          therefore should be considered a developer aid, not an audit log.
        """
        print("\n" + "=" * 60)
        print(f"Multi-Agent Orchestrator Trace (Objective: {self.objective})")
        print("=" * 60)
        history = self._history if all_turns else [self._history[-1]]
        for turn_idx, turn in enumerate(history):
            print(f"\n[Turn {turn_idx + 1}]")
            print("-" * 50)
            for role, text in turn:
                header = f"[{role.upper()}]"
                print(f"{header}:\n" + "-" * len(header))
                for line in textwrap.wrap(text, width=96):
                    print(line)
                print()
        print("=" * 60)

    def _validate_context(self, *, context: MultiTurnAttackContext) -> None:
        """Ensure configuration is valid before execution."""
        if not (context.objective or self.objective):
            raise ValueError("An objective must be provided for the attack.")
        if not self.agent_chain:
            raise ValueError("agent_chain must contain at least one AgentEntry.")

    async def _teardown_async(self, *, context: MultiTurnAttackContext) -> None:
        """Hook for cleanup; currently a no-op."""
        pass
