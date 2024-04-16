# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod


class Identifier:

    @abstractmethod
    def to_identifier(self) -> dict[str, str]:
        pass

    def __str__(self) -> str:
        return f"{self.to_identifier}"
