# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from abc import abstractmethod


class Authenticator(abc.ABC):
    token: str

    @abstractmethod
    def refresh_token(self) -> str:
        raise NotImplementedError("refresh_token method not implemented")

    @abstractmethod
    def get_token(self) -> str:
        raise NotImplementedError("get_token method not implemented")
