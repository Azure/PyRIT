# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod


class Identifier:

    @abstractmethod
    def get_identifier(self) -> dict[str, str]:
        pass

    def __str__(self) -> str:
        return f"{self.get_identifier}"
