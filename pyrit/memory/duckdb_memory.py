# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Union, Optional
import logging

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.base import Engine
from contextlib import closing

from pyrit.memory.memory_models import EmbeddingData, PromptMemoryEntry, Base
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.common.path import RESULTS_PATH
from pyrit.common.singleton import Singleton
from pyrit.models import PromptRequestPiece

logger = logging.getLogger(__name__)


class DuckDBMemory(MemoryInterface, metaclass=Singleton):
    """
    A class to manage conversation memory using DuckDB as the backend database. It leverages SQLAlchemy Base models
    for creating tables and provides CRUD operations to interact with the tables.
    This class encapsulates the setup of the database connection, table creation based on SQLAlchemy models,
    and session management to perform database operations.
    """

    DEFAULT_DB_FILE_NAME = "pyrit_duckdb_storage.db"

    def __init__(
        self,
        *,
        db_path: Union[Path, str] = None,
        verbose: bool = False,
    ):
        super(DuckDBMemory, self).__init__()

        if db_path == ":memory:":
            self.db_path: Union[Path, str] = ":memory:"
        else:
            self.db_path = Path(db_path or Path(RESULTS_PATH, self.DEFAULT_DB_FILE_NAME)).resolve()

        self.engine = self._create_engine(has_echo=verbose)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

    def _create_engine(self, *, has_echo: bool) -> Engine:
        """Creates the SQLAlchemy engine for DuckDB.

        Creates an engine bound to the specified database file. The `has_echo` parameter
        controls the verbosity of SQL execution logging.

        Args:
            has_echo (bool): Flag to enable detailed SQL execution logging.
        """
        try:
            # Create the SQLAlchemy engine.
            engine = create_engine(f"duckdb:///{self.db_path}", echo=has_echo)
            logger.info(f"Engine created successfully for database: {self.db_path}")
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

    def get_all_prompt_entries(self) -> list[PromptMemoryEntry]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result = self.query_entries(PromptMemoryEntry)
        return result

    def get_all_embeddings(self) -> list[EmbeddingData]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result = self.query_entries(EmbeddingData)
        return result

    def get_prompt_entries_with_conversation_id(self, *, conversation_id: str) -> list[PromptMemoryEntry]:
        """
        Retrieves a list of ConversationData objects that have the specified conversation ID.

        Args:
            conversation_id (str): The conversation ID to filter the table.

        Returns:
            list[ConversationData]: A list of ConversationData objects matching the specified conversation ID.
        """
        try:
            return self.query_entries(
                PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == conversation_id
            )
        except Exception as e:
            logger.exception(f"Failed to retrieve conversation_id {conversation_id} with error {e}")
            return []

    def get_prompt_entries_by_orchestrator(self, *, orchestrator_id: int) -> list[PromptMemoryEntry]:
        """
        Retrieves a list of PromptMemoryEntry objects that have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            list[PromptMemoryEntry]: A list of PromptMemoryEntry objects matching the specified orchestrator ID.
        """
        try:
            return self.query_entries(
                PromptMemoryEntry,
                conditions=PromptMemoryEntry.orchestrator_identifier.op("->>")("id") == str(orchestrator_id),
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error: Failed to retrieve ConversationData with orchestrator {orchestrator_id}. {e}"
            )
            return []

    def add_request_pieces_to_memory(self, *, request_pieces: list[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.
        If necessary, generates embedding data for applicable entries

        Args:
            entries (list[Base]): The list of database model instances to be inserted.
        """
        embedding_entries = []

        if self.memory_embedding:
            for piece in request_pieces:
                embedding_entry = self.memory_embedding.generate_embedding_memory_data(prompt_request_piece=piece)
                embedding_entries.append(embedding_entry)

        # The ordering of this is weird because after memory entries are inserted, we lose the reference to them
        # and also entries must be inserted before embeddings because of the foreign key constraint
        self._insert_entries(entries=[PromptMemoryEntry(entry=piece) for piece in request_pieces])

        if embedding_entries:
            self._insert_entries(entries=embedding_entries)

    def update_entries_by_conversation_id(self, *, conversation_id: str, update_fields: dict) -> bool:
        """
        Updates entries for a given conversation ID with the specified field values.

        Args:
            conversation_id (str): The conversation ID of the entries to be updated.
            update_fields (dict): A dictionary of field names and their new values.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        # Fetch the relevant entries using query_entries
        entries_to_update = self.query_entries(
            PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == conversation_id
        )

        # Check if there are entries to update
        if not entries_to_update:
            logger.info(f"No entries found with conversation_id {conversation_id} to update.")
            return False

        # Use the utility function to update the entries
        return self.update_entries(entries=entries_to_update, update_fields=update_fields)

    def get_all_table_models(self) -> list[Base]:  # type: ignore
        """
        Returns a list of all table models used in the database by inspecting the Base registry.

        Returns:
            list[Base]: A list of SQLAlchemy model classes.
        """
        # The '__subclasses__()' method returns a list of all subclasses of Base, which includes table models
        return Base.__subclasses__()

    def get_session(self) -> Session:
        """
        Provides a session for database operations.
        """
        return self.SessionFactory()

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

    def update_entries(self, *, entries: list[Base], update_fields: dict) -> bool:  # type: ignore
        """
        Updates the given entries with the specified field values.

        Args:
            entries (list[Base]): A list of SQLAlchemy model instances to be updated.
            update_fields (dict): A dictionary of field names and their new values.
        """
        with closing(self.get_session()) as session:
            try:
                for entry in entries:
                    # Ensure the entry is attached to the session. If it's detached, merge it.
                    entry_in_session = session.merge(entry)
                    for field, value in update_fields.items():
                        setattr(entry_in_session, field, value)
                session.commit()
                return True
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error updating entries: {e}")
                return False

    def export_all_tables(self, *, export_type: str = "json"):
        """
        Exports all table data using the specified exporter.

        Iterates over all tables, retrieves their data, and exports each to a file named after the table.

        Args:
            export_type (str): The format to export the data in (defaults to "json").
        """
        table_models = self.get_all_table_models()

        for model in table_models:
            data = self.query_entries(model)
            table_name = model.__tablename__
            file_extension = f".{export_type}"
            file_path = RESULTS_PATH / f"{table_name}{file_extension}"
            self.exporter.export_data(data, file_path=file_path, export_type=export_type)

    def print_schema(self):
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        for table_name in metadata.tables:
            table = metadata.tables[table_name]
            print(f"Schema for {table_name}:")
            for column in table.columns:
                print(f"  Column {column.name} ({column.type})")

    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Engine disposed successfully.")

    def reset_database(self):
        """Drop and recreate existing tables"""
        # Drop all existing tables
        Base.metadata.drop_all(self.engine)
        # Recreate the tables
        Base.metadata.create_all(self.engine, checkfirst=True)
