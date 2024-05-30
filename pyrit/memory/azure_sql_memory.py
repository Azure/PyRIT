# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal
import struct
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.base import Engine

from azure.identity import DefaultAzureCredential

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
    TOKEN_URL = "https://database.windows.net/"  # The token URL for any Azure SQL database
    TIMEOUT_SECONDS = 60

    def __init__(
        self,
        *,
        server: str,
        database: str,
        driver_version: Literal['17', '18'] = '18',
        timeout: int = TIMEOUT_SECONDS,
        verbose: bool = False
    ):
        super(AzureSQLMemory, self).__init__()

        self._server = server
        self._database = database
        self._timeout = timeout

        if driver_version == '17':
            self._driver = 'ODBC+Driver+17+for+SQL+Server'
        elif driver_version == '18':
            self._driver = 'ODBC+Driver+18+for+SQL+Server'
        else:
            raise ValueError(f"Unsupported driver version: {driver_version}")

        self.engine = self._create_engine(has_echo=verbose)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

    def _create_engine(self, *, has_echo: bool) -> Engine:
        """Creates the SQLAlchemy engine for Azure SQL Server.

        Creates an engine bound to the specified server and database. The `has_echo` parameter
        controls the verbosity of SQL execution logging.

        Args:
            has_echo (bool): Flag to enable detailed SQL execution logging.
        """

        azure_credentials = DefaultAzureCredential(process_timeout=self._timeout)

        try:
            # Create the SQLAlchemy engine.
            engine = create_engine(f"mssql+pyodbc://@{self._server}/{self._database}?driver={self._driver}", echo=has_echo)
        except SQLAlchemyError as e:
            logger.exception(f"Error creating the engine for the database: {e}")
            raise
        else:
            logger.info(f"Engine created successfully for database: {self._server}/{self._database}?driver={self._driver}")
            @event.listens_for(engine, "do_connect")
            def provide_token(_dialect, _conn_rec, cargs, cparams):
                # remove the "Trusted_Connection" parameter that SQLAlchemy adds
                cargs[0] = cargs[0].replace(";Trusted_Connection=Yes", "")
                # create token credential
                azure_token = azure_credentials.get_token(self.TOKEN_URL)
                raw_token = azure_token.token.encode("utf-16-le")
                token_struct = struct.pack(f"<I{len(raw_token)}s", len(raw_token), raw_token)
                # apply it to keyword arguments
                cparams["attrs_before"] = {self.SQL_COPT_SS_ACCESS_TOKEN: token_struct}
            return engine

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
        return super()._add_embeddings_to_memory(embedding_data=embedding_data)

    def _get_prompt_pieces_by_orchestrator(self, *, orchestrator_id: int) -> list[PromptRequestPiece]:
        return super()._get_prompt_pieces_by_orchestrator(orchestrator_id=orchestrator_id)

    def _get_prompt_pieces_with_conversation_id(self, *, conversation_id: str) -> list[PromptRequestPiece]:
        return super()._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)

    def add_request_pieces_to_memory(self, *, request_pieces: list[PromptRequestPiece]) -> None:
        return super().add_request_pieces_to_memory(request_pieces=request_pieces)

    def add_scores_to_memory(self, *, scores: list[Score]) -> None:
        return super().add_scores_to_memory(scores=scores)

    def dispose_engine(self):
        return super().dispose_engine()

    def get_all_embeddings(self) -> list[EmbeddingData]:
        return super().get_all_embeddings()

    def get_all_prompt_pieces(self) -> list[PromptRequestPiece]:
        return super().get_all_prompt_pieces()

    def get_prompt_request_pieces_by_id(self, *, prompt_ids: list[str]) -> list[PromptRequestPiece]:
        return super().get_prompt_request_pieces_by_id(prompt_ids=prompt_ids)

    def get_scores_by_prompt_ids(self, *, prompt_request_response_ids: list[str]) -> list[Score]:
        return super().get_scores_by_prompt_ids(prompt_request_response_ids=prompt_request_response_ids)
