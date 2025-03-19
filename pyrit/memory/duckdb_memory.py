# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from contextlib import closing
from pathlib import Path
from typing import MutableSequence, Optional, Sequence, TypeVar, Union

from sqlalchemy import MetaData, and_, create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.common.path import DB_DATA_PATH
from pyrit.common.singleton import Singleton
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.memory.memory_models import (
    Base,
    EmbeddingDataEntry,
    PromptMemoryEntry,
    SeedPromptEntry,
)
from pyrit.models import DiskStorageIO, PromptRequestPiece

logger = logging.getLogger(__name__)

Model = TypeVar("Model")


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
            self.db_path = Path(db_path or Path(DB_DATA_PATH, self.DEFAULT_DB_FILE_NAME)).resolve()
        self.results_path = str(DB_DATA_PATH)

        self.engine = self._create_engine(has_echo=verbose)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

    def _init_storage_io(self):
        # Handles disk-based storage for DuckDB local memory.
        self.results_storage_io = DiskStorageIO()

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

    def get_all_embeddings(self) -> Sequence[EmbeddingDataEntry]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result: Sequence[EmbeddingDataEntry] = self._query_entries(EmbeddingDataEntry)
        return result

    def _get_prompt_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]):
        conditions = [PromptMemoryEntry.labels.op("->>")(key) == value for key, value in memory_labels.items()]
        return and_(*conditions)

    def _get_prompt_pieces_prompt_metadata_conditions(self, *, prompt_metadata):
        conditions = [
            PromptMemoryEntry.prompt_metadata.op("->>")(key) == value for key, value in prompt_metadata.items()
        ]
        return and_(*conditions)

    def _get_prompt_pieces_orchestrator_conditions(self, *, orchestrator_id: str):
        return PromptMemoryEntry.orchestrator_identifier.op("->>")("id") == orchestrator_id

    def _get_seed_prompts_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]):
        conditions = [SeedPromptEntry.prompt_metadata.op("->>")(key) == value for key, value in metadata.items()]
        return and_(*conditions)

    def add_request_pieces_to_memory(self, *, request_pieces: Sequence[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.

        """
        self._insert_entries(entries=[PromptMemoryEntry(entry=piece) for piece in request_pieces])

    def _add_embeddings_to_memory(self, *, embedding_data: Sequence[EmbeddingDataEntry]) -> None:
        """
        Inserts embedding data into memory storage
        """
        self._insert_entries(entries=embedding_data)

    def get_all_table_models(self) -> list[type[Base]]:  # type: ignore
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

    def _insert_entries(self, *, entries: Sequence[Base]) -> None:  # type: ignore
        """Inserts multiple entries into the database."""
        with closing(self.get_session()) as session:
            try:
                session.add_all(entries)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error inserting multiple entries into the table: {e}")
                raise

    def _query_entries(
        self, Model, *, conditions: Optional = None, distinct: bool = False, join_scores: bool = False  # type: ignore
    ) -> MutableSequence[Model]:
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (Optional).
            distinct: Flag to return distinct rows (default is False).
            join_scores: Flag to join the scores table (default is False).

        Returns:
            List of model instances representing the rows fetched from the table.
        """
        with closing(self.get_session()) as session:
            try:
                query = session.query(Model)
                if join_scores and Model == PromptMemoryEntry:
                    query = query.options(joinedload(PromptMemoryEntry.scores))
                if conditions is not None:
                    query = query.filter(conditions)
                if distinct:
                    return query.distinct().all()
                return query.all()
            except SQLAlchemyError as e:
                logger.exception(f"Error fetching data from table {Model.__tablename__}: {e}")
                return []

    def _update_entries(self, *, entries: MutableSequence[Base], update_fields: dict) -> bool:  # type: ignore
        """
        Updates the given entries with the specified field values.

        Args:
            entries (Sequence[Base]): A list of SQLAlchemy model instances to be updated.
            update_fields (dict): A dictionary of field names and their new values.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        if not update_fields:
            raise ValueError("update_fields must be provided to update prompt entries.")
        with closing(self.get_session()) as session:
            try:
                for entry in entries:
                    # Ensure the entry is attached to the session. If it's detached, merge it.
                    if not session.is_modified(entry):
                        entry_in_session = session.merge(entry)
                    else:
                        entry_in_session = entry
                    for field, value in update_fields.items():
                        if field in vars(entry_in_session):
                            setattr(entry_in_session, field, value)
                        else:
                            session.rollback()
                            raise ValueError(
                                f"Field '{field}' does not exist in the table \
                                            '{entry_in_session.__tablename__}'. Rolling back changes..."
                            )
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
            data: MutableSequence[Base] = self._query_entries(model)
            table_name = model.__tablename__
            file_extension = f".{export_type}"
            file_path = DB_DATA_PATH / f"{table_name}{file_extension}"
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
