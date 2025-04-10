import logging
from typing import List, Optional, Union

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.score import Score
from pyrit.score import Scorer

logger = logging.getLogger(__name__)

class ScoreEvaluator:
    """
    Evaluates scores for orchestrator responses.
    
    Responsible for:
    - Evaluating if objectives are achieved
    - Checking if responses are refused
    - Tracking score history
    - Providing score feedback
    """
    
    def __init__(
            self, 
            *, 
            scorer: Scorer, 
            use_score_as_feedback: bool = True, 
            objective_achieved_score_threshold: float = 0.5
        ):
        """
        Initialize the score evaluator.
        
        Args:
            scorer: The scorer to use for evaluation
        """
        self._scorer = scorer
        self._score_history: List[Score] = []
        self._use_score_as_feedback = use_score_as_feedback
        self._objective_achieved_score_threshold = objective_achieved_score_threshold
    
    async def evaluate_response(
        self, 
        *,
        response: Union[PromptRequestPiece, PromptRequestResponse],
        only_allowed_type: Optional[str] = None,
        store_history: bool = True
    ) -> Score:
        """
        Evaluate a response using the configured scorer.
        
        Args:
            response: The response to evaluate
            store_history: Whether to store the score in history
            
        Returns:
            The score result
            
        Raises:
            ValueError: If the scorer returns an unexpected score type
        """
        # If we got a full response, extract the first piece
        if isinstance(response, PromptRequestResponse):
            piece = response.get_piece()
        else:
            piece = response
        
        # Score the response
        score_result = (await self._scorer.score_async(request_response=piece))[0]

        # Check if the score type is valid
        if only_allowed_type and score_result.score_type != only_allowed_type:
            raise ValueError(f"The scorer must return a {only_allowed_type} score. The score type is {score_result.score_type}.")
        
        # Store in history if requested
        if store_history:
            self._score_history.append(score_result)
        
        return score_result
    
    def is_objective_achieved(
        self, 
        score: Optional[Score] = None
    ) -> bool:
        """
        Determine if the objective is achieved based on score.
        
        Args:
            score: The score to check, or the latest score if None
            
        Returns:
            True if the objective is achieved, False otherwise
        """
        score_to_check = score or (self._score_history[-1] if self._score_history else None)
        
        if not score_to_check:
            return False
        
        # Check score type and value
        if score_to_check.score_type == "true_false":
            return bool(score_to_check.get_value())
        
        # For numeric scores, consider values >= 0.5 as successful
        elif score_to_check.score_type == "numeric":
            value = float(score_to_check.get_value())
            return value >= self._objective_achieved_score_threshold
        
        # For other score types, assume not achieved
        return False
    
    def get_feedback(self, score: Optional[Score] = None) -> Optional[str]:
        """
        Get feedback from a score for use in future prompts.
        
        Args:
            score: The score to get feedback from, or the latest score if None
            
        Returns:
            Feedback string, or None if no suitable feedback exists
        """
        if not self._use_score_as_feedback:
            return None

        score_to_check = score or (self._score_history[-1] if self._score_history else None)
        
        if not score_to_check:
            return None
            
        return score_to_check.score_rationale
    
    def add(self, score: Score) -> None:
        """Add a score to the history."""
        if score:
            self._score_history.append(score)
    
    @property
    def scorer_type(self) -> str:
        """Get the type of the scorer."""
        return self._scorer.get_identifier()["__type__"]
    
    def clear_history(self) -> None:
        """Clear the score history."""
        self._score_history = []
    
    @property
    def score_history(self) -> List[Score]:
        """Get the score history."""
        return self._score_history.copy()