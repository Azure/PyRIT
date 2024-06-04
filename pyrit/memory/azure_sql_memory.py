# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import struct
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.base import Engine

from pyrit.memory.memory_models import EmbeddingData, Base
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.common.singleton import Singleton
from pyrit.models import PromptRequestPiece
from pyrit.models import Score

logger = logging.getLogger(__name__)

class AzureSQLMemory(MemoryInterface, metaclass=Singleton):
    """
    A class to manage conversation memory using Azure SQL Server as the backend database. It leverages SQLAlchemy Base
    models for creating tables and provides CRUD operations to interact with the tables.

    This class encapsulates the setup of the database connection, table creation based on SQLAlchemy models,
    and session management to perform database operations.
    """

    SQL_COPT_SS_ACCESS_TOKEN = 1256  # Connection option for access tokens, as defined in msodbcsql.h

    def __init__(
        self,
        *,
        connection_string: str,
        auth_token: str = '',
        verbose: bool = False
    ):
        super(AzureSQLMemory, self).__init__()

        self._connection_string = connection_string
        self._auth_token = auth_token

        self.engine = self._create_engine(has_echo=verbose)

        if auth_token:
            self._enable_azure_authorization()

        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

    def _create_engine(self, *, has_echo: bool) -> Engine:
        """Creates the SQLAlchemy engine for Azure SQL Server.

        Creates an engine bound to the specified server and database. The `has_echo` parameter
        controls the verbosity of SQL execution logging.

        Args:
            has_echo (bool): Flag to enable detailed SQL execution logging.
        """

        try:
            # Create the SQLAlchemy engine.
            engine = create_engine(self._connection_string, echo=has_echo)
        except SQLAlchemyError as e:
            logger.exception(f"Error creating the engine for the database: {e}")
            raise
        else:
            logger.info(f"Engine created successfully for database: {self._server}/{self._database}?driver={self._driver}")
            return engine

    def _enable_azure_authorization(self) -> None:
        # TODO: investigate integrating this with the `azure_auth` module, enabling us to use refresh tokens, etc.

        @event.listens_for(self.engine, "do_connect")
        def provide_token(_dialect, _conn_rec, cargs, cparams):
            # remove the "Trusted_Connection" parameter that SQLAlchemy adds
            cargs[0] = cargs[0].replace(";Trusted_Connection=Yes", "")

            # encode the token
            azure_token = self._auth_token
            azure_token_bytes = azure_token.encode("utf-16-le")
            packed_azure_token = struct.pack(f"<I{len(azure_token_bytes)}s", len(azure_token_bytes), azure_token_bytes)

            # add the encoded token
            cparams["attrs_before"] = {self.SQL_COPT_SS_ACCESS_TOKEN: packed_azure_token}

    def _create_tables_if_not_exist(self):
        """
        Creates all tables defined in the Base metadata, if they don't already exist in the database.

        Raises:
            Exception: If there's an issue creating the tables in the database.
        """
        try:
            # Using the 'checkfirst=True' parameter to avoid attempting to recreate existing tables
            Base.metadata.create_all(self.engine, checkfirst=True)
        except Exception as e:
            logger.error(f"Error during table creation: {e}")

    def _add_embeddings_to_memory(self, *, embedding_data: list[EmbeddingData]) -> None:
        raise NotImplementedError("add_embeddings_to_memory method not implemented")

    def _get_prompt_pieces_by_orchestrator(self, *, orchestrator_id: int) -> list[PromptRequestPiece]:
        raise NotImplementedError("get_prompt_pieces_by_orchestrator method not implemented")

    def _get_prompt_pieces_with_conversation_id(self, *, conversation_id: str) -> list[PromptRequestPiece]:
        raise NotImplementedError("get_prompt_pieces_with_conversation_id method not implemented")

    def add_request_pieces_to_memory(self, *, request_pieces: list[PromptRequestPiece]) -> None:
        raise NotImplementedError("add_request_pieces_to_memory method not implemented")

    def add_scores_to_memory(self, *, scores: list[Score]) -> None:
        raise NotImplementedError("add_scores_to_memory method not implemented")

    def dispose_engine(self):
        raise NotImplementedError("dispose_engine method not implemented")

    def get_all_embeddings(self) -> list[EmbeddingData]:
        raise NotImplementedError("get_all_embeddings method not implemented")

    def get_all_prompt_pieces(self) -> list[PromptRequestPiece]:
        raise NotImplementedError("get_all_prompt_pieces method not implemented")

    def get_prompt_request_pieces_by_id(self, *, prompt_ids: list[str]) -> list[PromptRequestPiece]:
        raise NotImplementedError("get_prompt_request_pieces_by_id method not implemented")

    def get_scores_by_prompt_ids(self, *, prompt_request_response_ids: list[str]) -> list[Score]:
        raise NotImplementedError("get_scores_by_prompt_ids method not implemented")
