# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class PyritException(Exception, ABC):
    
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Status Code: {status_code}, Message: {message}")
    
    def process_exception(self) -> str:
        """
        Logs and returns a string representation of the exception.
        """
        log_message = f"{self.__class__.__name__} encountered: Status Code: {self.status_code}, Message: {self.message}"
        logger.error(log_message)
        # Return a string representation of the exception so users can extract and parse
        return json.dumps({"status_code": self.status_code, "message": self.message})
    

class BadRequestException(PyritException):
    """Exception class for bad client requests."""
    
    def __init__(self, status_code: int = 400, *, message: str = "Bad Request"):
        super().__init__(status_code, message)
        

class RateLimitException(PyritException):
    """Exception class for authentication errors."""
    
    def __init__(self, status_code: int = 429, *, message: str = "Rate Limit Exception"):
        super().__init__(status_code, message)
        

class EmptyResponseException(BadRequestException):
    """Exception class for empty response errors."""
    
    def __init__(self, status_code: int = 204, *, message: str = "No Content"):
        super().__init__(status_code=status_code, message=message)


