# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Custom Crescendo Attack that scores whole conversations but runs for max turns
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.models import PromptRequestResponse

class ConversationScoringAttack(CrescendoAttack):
    """
    Custom Crescendo Attack that sends the entire conversation history to the scorer 
    instead of just the latest response.
    """
    
    async def is_objective_achieved(self, prompt_request_response: PromptRequestResponse) -> bool:
        """
        Override the objective scoring to send entire conversation history to scorer.
        Returns True when objective is achieved (high score), False otherwise.
        """
        if not self._attack_scoring_config or not self._attack_scoring_config.objective_scorer:
            return False
        
        # Build the full conversation text from all exchanges
        conversation_text = ""
        
        # Get all prompt request responses for this conversation
        conversation_id = prompt_request_response.request_pieces[0].conversation_id
        all_responses = self._memory.get_prompt_request_responses(conversation_id=conversation_id)
        
        # Sort by timestamp to get chronological order
        all_responses.sort(key=lambda x: x.request_pieces[0].timestamp)
        
        # Build conversation text
        for response in all_responses:
            for piece in response.request_pieces:
                if piece.role == "user":
                    conversation_text += f"User: {piece.converted_value}\n"
                elif piece.role == "assistant":
                    conversation_text += f"Assistant: {piece.converted_value}\n"
        
        # Score the entire conversation
        scores = await self._attack_scoring_config.objective_scorer.score_text_async(
            text=conversation_text,
            task="evaluate entire conversation"
        )
        
        if scores:
            score = scores[0]
            
            # Handle true_false scores
            if score.score_type == "true_false":
                return bool(score.get_value())
            
            # Handle numeric scores
            if score.score_type == "float_scale":
                return float(score.get_value()) >= self._successful_objective_threshold
            
            # For other score types, assume not achieved
            return False
        
        return False
