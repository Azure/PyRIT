# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Protocol, Self

class Duplicable(Protocol):
    """
    Interface for objects that can be duplicated (deep copied).
    """
    
    def duplicate(self) -> Self:
        """
        Create a deep copy of this object.
        Supporting this interface implies that the object is changing parent classes
        or containers repeatedly.
        """
        return deepcopy(self)
    
    
class Serializable(Protocol):
    """
    Interface for objects that need to be serialized/deserialized.
    """
    
    def serialize(self) -> dict:
        """
        Serialize this object to a dictionary.
        """
        raise NotImplementedError("serialize method not implemented")
    
    @classmethod
    def deserialize(cls, data: dict) -> Self:
        """
        Deserialize this object from a dictionary.
        """
        raise NotImplementedError("deserialize method not implemented")