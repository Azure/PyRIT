# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import MutableSequence, Optional, Sequence, TypeVar, Union

from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.common.path import DB_DATA_PATH
from pyrit.common.singleton import Singleton
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.memory.memory_models import (
    AttackResultEntry,
    Base,
    EmbeddingDataEntry,
    PromptMemoryEntry,
)
from pyrit.models import DiskStorageIO, PromptRequestPiece

logger = logging.getLogger(__name__)

Model = TypeVar("Model")


class SQLiteMemory(MemoryInterface, metaclass=Singleton):
    """
    A memory interface that uses SQLite as the backend database.

    This class provides functionality to insert, query, and manage conversation data
    using SQLite. It supports both file-based and in-memory databases.

    Note: this is replacing the old DuckDB implementation.
    """

    DEFAULT_DB_FILE_NAME = "pyrit.db"

    def __init__(
        self,
        *,
        db_path: Optional[Union[Path, str]] = None,
        verbose: bool = False,
    ):
        super(SQLiteMemory, self).__init__()

        if db_path == ":memory:":
            self.db_path: Union[Path, str] = ":memory:"
        else:
            self.db_path = Path(db_path or Path(DB_DATA_PATH, self.DEFAULT_DB_FILE_NAME)).resolve()
        self.results_path = str(DB_DATA_PATH)

        self.engine = self._create_engine(has_echo=verbose)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

    def _init_storage_io(self):
        # Handles disk-based storage for SQLite local memory.
        self.results_storage_io = DiskStorageIO()

    def _create_engine(self, *, has_echo: bool) -> Engine:
        """Creates the SQLAlchemy engine for SQLite.

        Creates an engine bound to the specified database file. The `has_echo` parameter
        controls the verbosity of SQL execution logging.

        Args:
            has_echo (bool): Flag to enable detailed SQL execution logging.
        """
        try:
            # Create the SQLAlchemy engine.
            engine = create_engine(f"sqlite:///{self.db_path}", echo=has_echo)
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

    def _get_prompt_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]) -> list:
        """
        Generates SQLAlchemy filter conditions for filtering conversation pieces by memory labels.
        For SQLite, we use JSON_EXTRACT function to handle JSON fields.
        """
        # For SQLite, we use JSON_EXTRACT with text() and bindparams similar to Azure SQL approach
        json_conditions = " AND ".join([f"JSON_EXTRACT(labels, '$.{key}') = :{key}" for key in memory_labels])

        # Create SQL condition using SQLAlchemy's text() with bindparams
        # for safe parameter passing, preventing SQL injection
        condition = text(json_conditions).bindparams(**{key: str(value) for key, value in memory_labels.items()})
        return [condition]

    def _get_prompt_pieces_prompt_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]) -> list:
        """
        Generates SQLAlchemy filter conditions for filtering conversation pieces by prompt metadata.
        """
        json_conditions = " AND ".join(
            [f"JSON_EXTRACT(prompt_metadata, '$.{key}') = :{key}" for key in prompt_metadata]
        )

        # Create SQL condition using SQLAlchemy's text() with bindparams
        condition = text(json_conditions).bindparams(**{key: str(value) for key, value in prompt_metadata.items()})
        return [condition]

    def _get_prompt_pieces_attack_conditions(self, *, attack_id: str):
        """
        Generates SQLAlchemy filter conditions for filtering by attack ID.
        """
        return text("JSON_EXTRACT(attack_identifier, '$.id') = :attack_id").bindparams(attack_id=str(attack_id))

    def _get_seed_prompts_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]):
        """
        Generates SQLAlchemy filter conditions for filtering seed prompts by metadata.
        """
        json_conditions = " AND ".join([f"JSON_EXTRACT(prompt_metadata, '$.{key}') = :{key}" for key in metadata])

        # Create SQL condition using SQLAlchemy's text() with bindparams
        return text(json_conditions).bindparams(**{key: str(value) for key, value in metadata.items()})

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

    def _query_entries(
        self, Model, *, conditions: Optional = None, distinct: bool = False, join_scores: bool = False  # type: ignore
    ) -> MutableSequence[Model]:
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            Model: The SQLAlchemy model class corresponding to the table you want to query.
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
                elif Model == AttackResultEntry:
                    query = query.options(
                        joinedload(AttackResultEntry.last_response).joinedload(PromptMemoryEntry.scores),
                        joinedload(AttackResultEntry.last_score),
                    )
                if conditions is not None:
                    query = query.filter(conditions)
                if distinct:
                    return query.distinct().all()
                return query.all()
            except SQLAlchemyError as e:
                logger.exception(f"Error fetching data from table {Model.__tablename__}: {e}")
                return []

    def _insert_entry(self, entry: Base) -> None:  # type: ignore
        """
        Inserts an entry into the Table.

        Args:
            entry: An instance of a SQLAlchemy model to be inserted into the database.
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
                                f"Field '{field}' does not exist in the table '{entry_in_session.__tablename__}'. "
                                f"Rolling back changes..."
                            )
                session.commit()
                return True
            except SQLAlchemyError as e:
                session.rollback()
                logger.exception(f"Error updating entries: {e}")
                return False

    def get_session(self) -> Session:
        """
        Provides a SQLAlchemy session for transactional operations.

        Returns:
            Session: A SQLAlchemy session bound to the engine.
        """
        return self.SessionFactory()

    def reset_database(self) -> None:
        """
        Drops and recreates all tables in the database.
        """
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def dispose_engine(self) -> None:
        """
        Dispose the engine and close all connections.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Engine disposed and all connections closed.")

    def export_conversations(
        self,
        *,
        attack_id: Optional[str | uuid.UUID] = None,
        conversation_id: Optional[str | uuid.UUID] = None,
        prompt_ids: Optional[Sequence[str] | Sequence[uuid.UUID]] = None,
        labels: Optional[dict[str, str]] = None,
        sent_after: Optional[datetime] = None,
        sent_before: Optional[datetime] = None,
        original_values: Optional[Sequence[str]] = None,
        converted_values: Optional[Sequence[str]] = None,
        data_type: Optional[str] = None,
        not_data_type: Optional[str] = None,
        converted_value_sha256: Optional[Sequence[str]] = None,
        file_path: Optional[Path] = None,
        export_type: str = "json",
    ) -> Path:
        """
        Exports conversations and their associated scores from the database to a specified file.
        """
        # Import here to avoid circular import issues
        from pyrit.memory.memory_exporter import MemoryExporter

        if not self.exporter:
            self.exporter = MemoryExporter()

        # Get prompt pieces using the parent class method with appropriate filters
        prompt_pieces = self.get_prompt_request_pieces(
            attack_id=attack_id,
            conversation_id=conversation_id,
            prompt_ids=prompt_ids,
            labels=labels,
            sent_after=sent_after,
            sent_before=sent_before,
            original_values=original_values,
            converted_values=converted_values,
            data_type=data_type,
            not_data_type=not_data_type,
            converted_value_sha256=converted_value_sha256,
        )

        # Create the filename if not provided
        if not file_path:
            if attack_id:
                file_name = f"{attack_id}.{export_type}"
            elif conversation_id:
                file_name = f"{conversation_id}.{export_type}"
            else:
                file_name = f"all_conversations.{export_type}"
            file_path = Path(DB_DATA_PATH, file_name)

        # Get scores for the prompt pieces
        if prompt_pieces:
            prompt_request_response_ids = [str(piece.id) for piece in prompt_pieces]
            scores = self.get_prompt_scores(prompt_ids=prompt_request_response_ids)
        else:
            scores = []

        # Merge conversations and scores - create the data structure manually
        merged_data = []
        for piece in prompt_pieces:
            piece_data = piece.to_dict()
            # Find associated scores
            piece_scores = [score for score in scores if score.prompt_request_response_id == piece.id]
            piece_data["scores"] = [score.to_dict() for score in piece_scores]
            merged_data.append(piece_data)

        # Export to JSON manually since the exporter expects objects but we have dicts
        with open(file_path, "w") as f:
            import json

            json.dump(merged_data, f, indent=4)
        return file_path

    def print_schema(self):
        """
        Prints the schema of all tables in the SQLite database.
        """
        print("Database Schema:")
        print("================")
        for table_name, table in Base.metadata.tables.items():
            print(f"\nTable: {table_name}")
            print("-" * (len(table_name) + 7))  # +7 to align to be under header ("table: " is 7 chars)
            for column in table.columns:
                nullable = "NULL" if column.nullable else "NOT NULL"
                default = f" DEFAULT {column.default}" if column.default else ""
                print(f"  {column.name}: {column.type} {nullable}{default}")

    def export_all_tables(self, *, export_type: str = "json"):
        """
        Exports all table data using the specified exporter.

        Iterates over all tables, retrieves their data, and exports each to a file named after the table.

        Args:
            export_type (str): The format to export the data in (defaults to "json").
        """
        table_models = self.get_all_table_models()

        for model in table_models:
            data = self._query_entries(model)  # type: ignore
            table_name = model.__tablename__
            file_extension = f".{export_type}"
            file_path = DB_DATA_PATH / f"{table_name}{file_extension}"
            # Convert to list for exporter compatibility
            self.exporter.export_data(list(data), file_path=file_path, export_type=export_type)
