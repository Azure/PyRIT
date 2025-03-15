import abc
import uuid
from datetime import datetime
from typing import Optional, Dict


class AttackConfiguration(abc.ABC):
    """
    Represents an Attack Configuration
    Parameters:
        id (UUID): The unique identifier for the attack configuration.
        orchestrator_identifier (Dict[str, str]): The orchestrator identifier of the orchestrator initiating the attack.
        conversation_objective (str): The objective of the attack.
        attack_result (Dict[str, str]): The result of the attack.
        labels (Dict[str, str]): The labels associated with the labels.
        start_time (datetime): The start timestamp of the attack.
        end_time (datetime): The end timestamp of the attack.
    """

    def __init__(
        self,
        *,
        id: Optional[uuid.UUID | str] = None,
        orchestrator_identifier: Optional[Dict[str, str]] = None,
        conversation_objective: Optional[str] = None,
        attack_result: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        self.id = id if id else uuid.uuid4()
        self.orchestrator_identifier = orchestrator_identifier or {}
        self.conversation_objective = conversation_objective
        self.attack_result = attack_result or {}
        self.labels = labels or {}
        self.start_time = start_time if start_time else datetime.now()
        self.end_time = end_time

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "orchestrator_identifier": self.orchestrator_identifier,
            "conversation_objective": self.conversation_objective,
            "attack_result": self.attack_result,
            "labels": self.labels,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def __str__(self):
        return f"{self.id}: {self.orchestrator_identifier}: {self.conversation_objective}: {self.attack_result}: {self.labels}: {self.start_time}: {self.end_time}"

    __repr__ = __str__

    def __eq__(self, other) -> bool:
        return (
            self.id == other.id
            and self.orchestrator_identifier == other.orchestrator_identifier
            and self.conversation_objective == other.conversation_objective
            and self.attack_result == other.attack_result
            and self.labels == other.labels
            and self.start_time == other.start_time
            and self.end_time == other.end_time
        )