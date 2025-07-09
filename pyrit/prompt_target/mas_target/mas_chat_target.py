import json
import uuid
import jinja2
from typing import List, Dict, Optional, TypedDict

from pyrit.models import (
    PromptRequestResponse,
    PromptRequestPiece,
    construct_response_from_request,
)
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.logger import logger
from pyrit.memory import CentralMemory


class AgentEntry(TypedDict):
    role: str
    agent: PromptChatTarget


class MulitAgentSystemChatTarget(PromptChatTarget):
    """
    A flexible "Multi-Agent-System" wrapper for red teaming LLMs.
    Allows an ordered chain of agents (e.g. [recon_agent, strategy_agent, red_team_agent])
    to iteratively refine an adversarial prompt.

    Each time 'send_prompt_async' is called:
      1. The orchestrator's prompt (seed or previous reply) is added to conversation history.
      2. The full history and any agent-specific context are fed to the first agent.
      3. Each agent processes the input, generates output, and that output is passed to the next agent.
      4. The output of the last agent is returned as the adversarial prompt for the orchestrator.

    You can use two agents (e.g., strategy_agent → red_team_agent), or three (e.g., recon_agent → strategy_agent → red_team_agent), or more.
    """

    def __init__(
        self,
        *,
        agent_chain: List[AgentEntry],
        agent_contexts: Optional[Dict[str, str]] = None,
        objective: str,
        system_prompts: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            agent_chain: Ordered list of AgentEntry dicts, each with keys:
                - "role": str (e.g., "recon_agent", "strategy_agent", "red_team_agent")
                - "agent": PromptChatTarget instance
            agent_contexts: (optional) dict mapping agent role to context string, e.g., recon summary.
            objective: The attack/test objective string (for prompt templating).
            system_prompts: (optional) dict mapping role to system prompt (can use '{objective}' template).
        """
        super().__init__()
        self.agent_chain = agent_chain
        self.agent_contexts = agent_contexts or {}
        self.objective = objective
        self.system_prompts = system_prompts or {}

        # Map each internal agent -> conversation_id
        self._conv_ids: Dict[PromptChatTarget, str] = {}

        # Ensure we only init agents once
        self._inited = False

        # History as list of {"role":..., "text": ...}
        self._history: List[Dict[str, str]] = []

        logger.info("Initialized MASChatTarget with agents: %s", [a['role'] for a in self.agent_chain])

    def _initialize_agents(self):
        logger.info("Setting system prompts for agents…")
        for item in self.agent_chain:
            role = item["role"]
            agent = item["agent"]
            if agent not in self._conv_ids:
                cid = str(uuid.uuid4())
                self._conv_ids[agent] = cid
                sys_prompt = self.system_prompts.get(role, "")  # type: ignore
                if sys_prompt:
                    # Allow {objective} template
                    sys_prompt = jinja2.Template(sys_prompt).render(objective=self.objective)
                    agent.set_system_prompt(system_prompt=sys_prompt, conversation_id=cid)
                logger.info("Agent '%s' system prompt set.", role)
        self._inited = True
        logger.info("All agents initialized.")

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse):
        """
        Orchestrates a single "turn" of adversarial prompt refinement.

        Args:
            prompt_request: contains exactly one PromptRequestPiece whose converted_value is
                            the orchestrator's current prompt (seed on turn 1, else the target's last reply or feedback).

        Returns:
            A PromptRequestResponse wrapping the final adversarial prompt to send to the real target.
        """
        if not self._inited:
            self._initialize_agents()

        piece = prompt_request.request_pieces[0]
        current_input = piece.converted_value
        is_first = len(self._history) == 0

        logger.info("Received input for this turn: %r", current_input)

        # Append the new input into our running history
        role = "orchestrator_seed" if is_first else "target_response"
        self._history.append({"role": role, "text": current_input})
        logger.info("Appended to history: %s: %r", role, current_input)

        for idx, item in enumerate(self.agent_chain):
            role = item["role"]
            agent = item["agent"]
            
            if role == "red_team_agent" and idx > 0:
                # Only pass the "strategy" field from last agent's output (JSON) to the red team agent
                last_output = self._history[-1]["text"] if self._history else ""
                try:
                    strat_json = json.loads(last_output)
                    full_input = strat_json.get("strategy", last_output)
                    logger.info(f"Red Team Agent input (strategy): {full_input!r}")
                except Exception as e:
                    logger.warning(f"Failed to parse strategy JSON, using raw: {e}")
                    full_input = last_output

                # Append what the red team agent is sending
                self._history.append({"role": role, "text": full_input})
                self._persist_message(role, full_input, self._conv_ids[agent])

                # Actually send this prompt to the target LLM as user
                req = PromptRequestResponse(request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        conversation_id=self._conv_ids[agent],
                        original_value=full_input,
                        converted_value=full_input,
                    )
                ])
                res = await agent.send_prompt_async(prompt_request=req)
                output_text = res.request_pieces[0].converted_value

                # Record the model's response
                self._history.append({"role": "target_response", "text": output_text})
                self._persist_message("target_response", output_text, self._conv_ids[agent])
                break  # After red_team_agent and target_response, we're done with this turn.

            else:
                # Compose the input for this agent
                input_strs = []
                if role in self.agent_contexts and self.agent_contexts[role]:
                    input_strs.append(f"[{role.upper()} CONTEXT]\n{self.agent_contexts[role]}\n")

                # Always append conversation so far (for context-aware agents)
                input_strs.append("\n".join(f"{h['role']}: {h['text']}" for h in self._history))
                full_input = "\n".join(input_strs).strip()

                req = PromptRequestResponse(request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        conversation_id=self._conv_ids[agent],
                        original_value=full_input,
                        converted_value=full_input,
                    )
                ])
                res = await agent.send_prompt_async(prompt_request=req)
                output_text = res.request_pieces[0].converted_value

                self._history.append({"role": role, "text": output_text})
                # Persist each agent message for full chain auditability
                self._persist_message(role, output_text, self._conv_ids[agent])

        final_entry = self._history[-1]
        final_prompt = final_entry["text"]
        logger.info("Sending to orchestrator: %r", final_prompt)
        logger.info("=== FULL MAS HISTORY ===\n%s",
                    json.dumps(self._history, indent=2, ensure_ascii=False))
        return construct_response_from_request(
            request=piece,
            response_text_pieces=[final_prompt],
        )
    
    def _persist_message(self, role: str, text: str, conversation_id: str):
        # Map MAS agent role to valid ChatMessageRole
        if role  in ("orchestrator_seed", "target_response"):
            chat_role = "user"
        else:
            chat_role = "assistant"

        prp = PromptRequestPiece(
            role=chat_role,
            original_value=text,
            converted_value=text,
            conversation_id=conversation_id,
        )
        CentralMemory.get_memory_instance().add_request_pieces_to_memory(request_pieces=[prp])

    def get_history(self):
        """Retrieve the full MAS agent chain (list of dicts with role/text)."""
        return self._history
    
    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """Ensure we only ever receive a single text prompt piece."""
        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("MASAdversarialChatTarget only supports text prompt input.")

    def is_json_response_supported(self) -> bool:
        """We parse JSON ourselves, so we don't request JSON from the target."""
        return False