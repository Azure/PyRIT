# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from contextlib import closing
from typing import Optional

from sqlalchemy import create_engine, func, and_
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.common.singleton import Singleton
from pyrit.memory.memory_models import EmbeddingData, Base, PromptMemoryEntry, ScoreEntry
from pyrit.memory.memory_interface import MemoryInterface
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

    def __init__(self, *, connection_string: str, verbose: bool = False):
        super(AzureSQLMemory, self).__init__()

        self._connection_string = connection_string

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

        try:
            # Create the SQLAlchemy engine.
            engine = create_engine(self._connection_string, echo=has_echo)
            logger.info(f"Engine created successfully for database: {engine.name}")
            return engine
        except SQLAlchemyError as e:
            logger.exception(f"Error creating the engine for the database: {e}")
            raise

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
        """
        Inserts embedding data into memory storage
        """
        self._insert_entries(entries=embedding_data)

    def _get_prompt_pieces_by_orchestrator(self, *, orchestrator_id: int) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPiece objects matching the specified orchestrator ID.
        """
        try:
            return self.query_entries(
                PromptMemoryEntry,
                conditions=and_(
                    func.ISJSON(PromptMemoryEntry.orchestrator_identifier) > 0,
                    func.JSON_VALUE(PromptMemoryEntry.orchestrator_identifier, "$.id") == str(orchestrator_id),
                ),
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error: Failed to retrieve ConversationData with orchestrator {orchestrator_id}. {e}"
            )
            return []

    def _get_prompt_pieces_with_conversation_id(self, *, conversation_id: str) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to match.

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPieces with the specified conversation ID.
        """
        try:
            return self.query_entries(
                PromptMemoryEntry,
                conditions=PromptMemoryEntry.conversation_id == conversation_id,
            )
        except Exception as e:
            logger.exception(f"Failed to retrieve conversation_id {conversation_id} with error {e}")
            return []

    def add_request_pieces_to_memory(self, *, request_pieces: list[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.

        """
        self._insert_entries(entries=[PromptMemoryEntry(entry=piece) for piece in request_pieces])

    def add_scores_to_memory(self, *, scores: list[Score]) -> None:
        """
        Inserts a list of scores into the memory storage.
        """
        self._insert_entries(entries=[ScoreEntry(entry=score) for score in scores])

    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Engine disposed successfully.")

    def get_all_embeddings(self) -> list[EmbeddingData]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result = self.query_entries(EmbeddingData)
        return result

    def get_all_prompt_pieces(self) -> list[PromptRequestPiece]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result: list[PromptMemoryEntry] = self.query_entries(PromptMemoryEntry)
        return [entry.get_prompt_request_piece() for entry in result]

    def get_prompt_request_pieces_by_id(self, *, prompt_ids: list[str]) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified prompt ids.

        Args:
            prompt_ids (list[str]): The prompt IDs to match.

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPiece with the specified conversation ID.
        """
        try:
            return self.query_entries(
                PromptMemoryEntry,
                conditions=PromptMemoryEntry.id.in_(prompt_ids),
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error: Failed to retrieve ConversationData with orchestrator {prompt_ids}. {e}"
            )
            return []

    def get_scores_by_prompt_ids(self, *, prompt_request_response_ids: list[str]) -> list[Score]:
        """
        Gets a list of scores based on prompt_request_response_ids.
        """
        entries = self.query_entries(
            ScoreEntry, conditions=ScoreEntry.prompt_request_response_id.in_(prompt_request_response_ids)
        )

        return [entry.get_score() for entry in entries]

    # The following methods are not part of MemoryInterface, but seem
    # common between SQLAlchemy-based implementations, regardless of engine.
    # Perhaps we should find a way to refactor
    def _insert_entries(self, *, entries: list[Base]) -> None:  # type: ignore
        """Inserts multiple entries into the database."""
        with closing(self.get_session()) as session:
            try:
                session.add_all(entries)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error inserting multiple entries into the table: {e}")
                raise

    def _insert_entry(self, entry: Base) -> None:  # type: ignore
        """
        Inserts an entry into the Table.

        Args:
            entry: An instance of a SQLAlchemy model to be added to the Table.
        """
        with closing(self.get_session()) as session:
            try:
                session.add(entry)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error inserting entry into the table: {e}")

    def get_session(self) -> Session:
        """
        Provides a session for database operations.
        """
        return self.SessionFactory()

    def query_entries(self, model, *, conditions: Optional = None) -> list[Base]:  # type: ignore
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (optional).

        Returns:
            List of model instances representing the rows fetched from the table.
        """
        with closing(self.get_session()) as session:
            try:
                query = session.query(model)
                if conditions is not None:
                    query = query.filter(conditions)
                return query.all()
            except SQLAlchemyError as e:
                logger.exception(f"Error fetching data from table {model.__tablename__}: {e}")

    def reset_database(self):
        """Drop and recreate existing tables"""
        # Drop all existing tables
        Base.metadata.drop_all(self.engine)
        # Recreate the tables
        Base.metadata.create_all(self.engine, checkfirst=True)
