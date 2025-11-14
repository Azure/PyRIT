# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DockerImageFile:
    """
    Dataclass for Docker images.
    Some helper methods for parsing and security purposes.
    """
    # This is mandatory, because without an image, you can't have a container.
    source_file: Path

    # This one is optional because you may or may not have these.
    image_contents: Optional[Path] = None
    
    def __post_init__(self) -> None:
        if not self._verify_dockerfile(self.source_file, self.image_contents):
            raise ValueError(f"The contents of image {self.source_file} could not be verified!")
    
    def _verify_dockerfile(self, source_file: Path, image_contents: Optional[Path]) -> bool:
        """
        Ensure that the dockerfile is well-formatted and safe.
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        """
        Convert to string for client.containers.run call (image parameter)
        """
        raise NotImplementedError
        

@dataclass(frozen=True)
class DockerComposeConfig:
    """
    Config for Docker Compose.
    Responsibility is securely creating container, not image or contents. 
    """
    
    # Source file takes priority if given when building DockerImage.
    # Note that this is for the dockerfile, NOT compose.
    source_file: Optional[Path] = None
    
    # If none is provided, least-privilege secure defaults are used.
    image: Optional[str] = None
    ports: Optional[Dict[str, Tuple[str, int]]] = None
    mcp_port: int = 8080
    network_mode: str = "none"
    mem_limit: Optional[str] = "512m"
    cpu_quota: Optional[int] = 50000
    cap_drop: tuple[str, ...] = ("ALL",)
    cap_add: tuple[str, ...] = ()
    read_only: bool = False
    
    def __post_init__(self) -> None:
        # If the source file and attributes are given, the explicit attributes are given precedence,
        # but those that are not listed take default values.
        fields = [attr for attr in list(self.__dataclass_fields__.values()) if attr != "source_file"]
        if self.source_file and any(fields):
            logger.warning("DockerImage created with source file and explicit attributes.")
        elif self.source_file:
            config = self._parse_config_file(self.source_file)
                
    def _parse_config_file(self, src: Path) -> Dict:
        with open(src) as f:
            return json.loads(f)
    
    
    def unpack(self) -> Dict:
        return {
            "image": self.image 
        }
        

@dataclass(frozen=True)
class CTF(DockerImageFile):
    raise NotImplementedError
