# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from abc import abstractmethod


class Authenticator(abc.ABC):
    """Abstract base class for authenticators."""

    token: str

    @abstractmethod
    def refresh_token(self) -> str:
        """
        Refresh the authentication token.

        Returns:
            str: The refreshed authentication token.
        """
        raise NotImplementedError("refresh_token method not implemented")

    @abstractmethod
    def get_token(self) -> str:
        """
        Get the current authentication token.

        Returns:
            str: The current authentication token.
        """
        raise NotImplementedError("get_token method not implemented")
