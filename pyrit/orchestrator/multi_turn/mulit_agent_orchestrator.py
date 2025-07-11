# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
import textwrap
from typing import List, Dict, Optional, TypedDict, Any

from pyrit.orchestrator import Orchestrator
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.memory import CentralMemory
from pyrit.score import Scorer

class AgentEntry(TypedDict):
    """A single entry in the agent chain, specifying the agent's role and instance."""
    role: str
    agent: PromptChatTarget

class MultiAgentOrchestrator(Orchestrator):
    """
    Orchestrator subclass: manages a chain of agents, per-agent context, and converters,
    and executes multi-step attacks on a target model, tracking every step.
    """

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
    ):
        """
        Initializes a MultiAgentOrchestrator.

        Args:
            agent_chain: List of agent entries with "role" and "agent" keys.
            agent_contexts: Optional dict mapping agent roles to context strings.
            agent_converters: Optional dict mapping agent roles to converter lists.
            objective: The test/attack objective string.
            system_prompts: Optional dict mapping agent role to system prompt template.
            prompt_converters: Optional list of converters (used globally if no per-agent mapping).
            objective_target: The target to attack with the final adversarial prompt.
            scorer: Optional Scorer to evaluate whether the objective is achieved.
            verbose: Enables extra logging if True.
        """
        super().__init__(prompt_converters=prompt_converters, verbose=verbose)
        self.agent_chain = agent_chain
        self.agent_contexts = agent_contexts or {}
        self.agent_converters = agent_converters or {}
        self.objective = objective
        self.system_prompts = system_prompts or {}
        self.objective_target = objective_target
        self.scorer = scorer
        self._conv_ids: Dict[PromptChatTarget, str] = {}
        self._history: List[List[tuple[str, str]]] = []  # Per-turn, per-agent
        self._inited = False

    def _initialize_agents(self):
        """Sets conversation IDs and system prompts for all agents in the chain."""
        for item in self.agent_chain:
            role = item["role"]
            agent = item["agent"]
            if agent not in self._conv_ids:
                cid = str(uuid.uuid4())
                self._conv_ids[agent] = cid
                sys_prompt = self.system_prompts.get(role, "")
                if sys_prompt:
                    agent.set_system_prompt(system_prompt=sys_prompt, conversation_id=cid)
        self._inited = True

    async def run_attack_async(self, initial_input: str, max_turns: int = 5) -> Dict[str, Any]:
        """
        Executes the full multi-turn attack workflow: agents chain and final target attack.

        Args:
            initial_input: The seed input prompt for the agent chain.
            max_turns: Maximum number of conversation turns (attack iterations).

        Returns:
            Dict with keys: 'achieved', 'score', 'history'.
        """
        if not self._inited:
            self._initialize_agents()
        self._history = []
        input_text = initial_input
        achieved = False
        last_score = None

        for turn in range(max_turns):
            turn_trace = []  # [(role, text), ...] for this turn
            internal_history = []
            current_input = input_text
            for idx, item in enumerate(self.agent_chain):
                role = item["role"]
                agent = item["agent"]

                # Compose per-agent context + history + current input
                input_strs = []
                if role in self.agent_contexts and self.agent_contexts[role]:
                    input_strs.append(f"[{role.upper()} CONTEXT]\n{self.agent_contexts[role]}\n")
                input_strs.append(
                    "\n".join(f"{h['role']}: {h['text']}" for h in internal_history) if internal_history else ""
                )
                input_strs.append(current_input)
                full_input = "\n".join(filter(None, input_strs)).strip()

                # Prefer per-agent converters, fallback to global
                converters = self.agent_converters.get(role, self._prompt_converters)
                for converter in converters:
                    result = await converter.convert_async(prompt=full_input)
                    full_input = result.output_text

                req = PromptRequestResponse(request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        conversation_id=self._conv_ids[agent],
                        original_value=full_input,
                        converted_value=full_input,
                        prompt_metadata={"mas_role": role, "agent_index": idx},
                    )
                ])
                res = await agent.send_prompt_async(prompt_request=req)
                output_text = res.request_pieces[0].converted_value
                internal_history.append({"role": role, "text": output_text})
                self._persist_message(role, output_text, self._conv_ids[agent])
                turn_trace.append((role, output_text))
                current_input = output_text

            # FINAL agent output: objective target
            final_prompt = internal_history[-1]["text"]
            target_req = PromptRequestResponse(request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id=str(uuid.uuid4()),
                    original_value=final_prompt,
                    converted_value=final_prompt,
                )
            ])
            target_res = await self.objective_target.send_prompt_async(prompt_request=target_req)
            target_output = target_res.request_pieces[0].converted_value

            CentralMemory.get_memory_instance().add_request_response_to_memory(request=target_res)
            turn_trace.append(("objective_target", target_output))
            self._history.append(turn_trace)

            # Score if scorer present, exit early if success
            if self.scorer:
                score = await self.scorer.score_async(request_response=target_res.request_pieces[0])
                last_score = score[0]
                if last_score.get_value():
                    achieved = True
                    break
            input_text = target_output

        return {
            "achieved": achieved,
            "score": last_score,
            "history": self._history,
        }

    def _persist_message(self, role: str, text: str, conversation_id: str):
        """Persist a message to central memory for full auditability."""
        prp = PromptRequestPiece(
            role="assistant" if role not in ("orchestrator_seed", "target_response") else "user",
            original_value=text,
            converted_value=text,
            conversation_id=conversation_id,
            prompt_metadata={"mas_role": role},
        )
        CentralMemory.get_memory_instance().add_request_pieces_to_memory(request_pieces=[prp])

    def get_history(self) -> List[List[tuple[str, str]]]:
        """
        Returns the full multi-turn, per-agent message history.

        Returns:
            List[List[tuple[str, str]]]: Each item is a list of (role, text) pairs for one turn.
        """
        return self._history

    def print_conversation(self, all_turns: bool = True):
        """
        Pretty print the full history of the multi-agent orchestrator run.

        Args:
            all_turns: If True, print all turns. If False, print only the last turn.
        """
        print("\n" + "=" * 60)
        print(f"Multi-Agent Orchestrator Trace (Objective: {self.objective})")
        print("=" * 60)
        history = self._history if all_turns else [self._history[-1]]
        for turn_idx, turn in enumerate(history):
            print(f"\n[Turn {turn_idx+1}]")
            print("-" * 50)
            for role, text in turn:
                header = f"[{role.upper()}]"
                print(f"{header}:\n" + "-"*len(header))
                for line in textwrap.wrap(text, width=96):
                    print(line)
                print()
        print("=" * 60)