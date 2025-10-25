# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
from contextlib import closing
from typing import Any, MutableSequence, Optional, Sequence, TypeVar, Union

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.auth.azure_auth import AzureAuth
from pyrit.common import default_values
from pyrit.common.singleton import Singleton
from pyrit.common.path import PYRIT_PATH
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.memory.memory_models import (
    AttackResultEntry,
    Base,
    EmbeddingDataEntry,
    PromptMemoryEntry,
)
from pyrit.models import MessagePiece, LocalFileStorageIO

logger = logging.getLogger(__name__)

Model = TypeVar("Model")


class AzureSQLMemory(MemoryInterface, metaclass=Singleton):
    """
    A class to manage conversation memory using a SQLAlchemy-compatible database as the backend.
    It can be configured to use various databases like PostgreSQL, SQL Server, or SQLite for on-premises deployments.

    This class encapsulates the setup of the database connection, table creation based on SQLAlchemy models,
    and session management to perform database operations.
    """

    # Default to a local SQLite database if no connection string is provided.
    # This is ideal for "on-site" or local-first development.
    DEFAULT_SQLITE_DB_PATH = os.path.join(PYRIT_PATH, "db", "pyrit.db")
    DEFAULT_CONNECTION_STRING = f"sqlite:///{DEFAULT_SQLITE_DB_PATH}"
    DB_CONNECTION_STRING = "DB_CONNECTION_STRING"

    def __init__(
        self,
        *,
        connection_string: Optional[str] = None,
        verbose: bool = False,
    ):
        self._connection_string = default_values.get_value(
            env_var_name=self.DB_CONNECTION_STRING, passed_value=connection_string, default_value=self.DEFAULT_CONNECTION_STRING
        )

        if self._connection_string.startswith("sqlite"):
            db_path = self._connection_string.split("///")[1]
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            logger.info(f"Using SQLite database at: {db_path}")

        self.engine = self._create_engine(has_echo=verbose)

        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

        super(AzureSQLMemory, self).__init__()

    def _init_storage_io(self):
        # For on-site, default to local file system storage.
        # The path can be configured via environment variables if needed.
        self.results_path = os.path.join(PYRIT_PATH, "results")
        os.makedirs(self.results_path, exist_ok=True)
        self.results_storage_io = LocalFileStorageIO(base_path=self.results_path)

    def _create_engine(self, *, has_echo: bool) -> Engine:
        """Creates the SQLAlchemy engine for Azure SQL Server.

        Creates an engine bound to the specified server and database. The `has_echo` parameter
        controls the verbosity of SQL execution logging.

        Args:
            has_echo (bool): Flag to enable detailed SQL execution logging.
        """

        try:
            # Create the SQLAlchemy engine.
            # Use pool_pre_ping (health check) to gracefully handle server-closed connections
            # by testing and replacing stale connections.
            # Set pool_recycle to 1800 seconds to prevent connections from being closed due to server timeout.

            engine = create_engine(self._connection_string, pool_recycle=1800, pool_pre_ping=True, echo=has_echo)
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

    def _add_embeddings_to_memory(self, *, embedding_data: Sequence[EmbeddingDataEntry]) -> None:
        """
        Inserts embedding data into memory storage
        """
        self._insert_entries(entries=embedding_data)

    def _get_message_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]) -> list:
        json_validation = "ISJSON(labels) = 1"
        json_conditions = " AND ".join([f"JSON_VALUE(labels, '$.{key}') = :{key}" for key in memory_labels])
        # Combine both conditions
        conditions = f"{json_validation} AND {json_conditions}"

        # Create SQL condition using SQLAlchemy's text() with bindparams
        # for safe parameter passing, preventing SQL injection
        condition = text(conditions).bindparams(**{key: str(value) for key, value in memory_labels.items()})
        return [condition]

    def _get_message_pieces_attack_conditions(self, *, attack_id: str) -> Any:
        return text("ISJSON(attack_identifier) = 1 AND JSON_VALUE(attack_identifier, '$.id') = :json_id").bindparams(
            json_id=str(attack_id)
        )

    def _get_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]):
        json_validation = "ISJSON(prompt_metadata) = 1"
        json_conditions = " AND ".join([f"JSON_VALUE(prompt_metadata, '$.{key}') = :{key}" for key in prompt_metadata])
        # Combine both conditions
        conditions = f"{json_validation} AND {json_conditions}"

        # Create SQL condition using SQLAlchemy's text() with bindparams
        # for safe parameter passing, preventing SQL injection
        condition = text(conditions).bindparams(**{key: str(value) for key, value in prompt_metadata.items()})
        return [condition]

    def _get_message_pieces_prompt_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]) -> list:
        return self._get_metadata_conditions(prompt_metadata=prompt_metadata)

    def _get_seed_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]) -> Any:
        return self._get_metadata_conditions(prompt_metadata=metadata)[0]

    def add_message_pieces_to_memory(self, *, message_pieces: Sequence[MessagePiece]) -> None:
        """
        Inserts a list of message pieces into the memory storage.

        """
        self._insert_entries(entries=[PromptMemoryEntry(entry=piece) for piece in message_pieces])

    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Engine disposed successfully.")

    def get_all_embeddings(self) -> Sequence[EmbeddingDataEntry]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result: Sequence[EmbeddingDataEntry] = self._query_entries(EmbeddingDataEntry)
        return result

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

    # The following methods are not part of MemoryInterface, but seem
    # common between SQLAlchemy-based implementations, regardless of engine.
    # Perhaps we should find a way to refactor
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

    def get_session(self) -> Session:
        """
        Provides a session for database operations.
        """
        return self.SessionFactory()

    def _query_entries(
        self,
        Model,
        *,
        conditions: Optional[Any] = None,  # type: ignore
        distinct: bool = False,
        join_scores: bool = False,
    ) -> MutableSequence[Model]:
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (Optional).
            distinct: Flag to return distinct rows (defaults to False).
            join_scores: Flag to join the scores table with entries (defaults to False).

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

    def reset_database(self):
        """Drop and recreate existing tables"""
        # Drop all existing tables
        Base.metadata.drop_all(self.engine)
        # Recreate the tables
        Base.metadata.create_all(self.engine, checkfirst=True)
