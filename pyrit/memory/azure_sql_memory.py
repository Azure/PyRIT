# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import struct

from contextlib import closing
from typing import MutableSequence, Optional, Sequence
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken

from sqlalchemy import create_engine, event, text, MetaData
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.common import default_values
from pyrit.common.singleton import Singleton
from pyrit.memory.memory_models import Base, EmbeddingDataEntry, PromptMemoryEntry, ScoreEntry
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import AzureBlobStorageIO, PromptRequestPiece, Score

logger = logging.getLogger(__name__)


class AzureSQLMemory(MemoryInterface, metaclass=Singleton):
    """
    A class to manage conversation memory using Azure SQL Server as the backend database. It leverages SQLAlchemy Base
    models for creating tables and provides CRUD operations to interact with the tables.

    This class encapsulates the setup of the database connection, table creation based on SQLAlchemy models,
    and session management to perform database operations.
    """

    SQL_COPT_SS_ACCESS_TOKEN = 1256  # Connection option for access tokens, as defined in msodbcsql.h
    TOKEN_URL = "https://database.windows.net/.default"  # The token URL for any Azure SQL database
    AZURE_SQL_DB_CONNECTION_STRING = "AZURE_SQL_DB_CONNECTION_STRING"
    AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL"
    SAS_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_RESULTS_SAS_TOKEN"

    def __init__(
        self,
        *,
        connection_string: Optional[str] = None,
        container_url: Optional[str] = None,
        sas_token: Optional[str] = None,
        verbose: bool = False,
    ):
        self._connection_string = default_values.get_required_value(
            env_var_name=self.AZURE_SQL_DB_CONNECTION_STRING, passed_value=connection_string
        )
        self._container_url: str = default_values.get_required_value(
            env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
        )
        try:
            self._sas_token: str = default_values.get_required_value(
                env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=sas_token
            )
        except ValueError:
            self._sas_token = None  # To use delegation SAS

        self.results_path = self._container_url

        self.engine = self._create_engine(has_echo=verbose)

        self._auth_token = self._create_auth_token()
        self._enable_azure_authorization()

        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

        super(AzureSQLMemory, self).__init__()

    def _init_storage_io(self):
        # Handle for Azure Blob Storage when using Azure SQL memory.
        self.storage_io = AzureBlobStorageIO(container_url=self._container_url, sas_token=self._sas_token)

    def _create_auth_token(self) -> AccessToken:
        azure_credentials = DefaultAzureCredential()
        return azure_credentials.get_token(self.TOKEN_URL)

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

    def _enable_azure_authorization(self) -> None:
        """
        The following is necessary because of how SQLAlchemy and PyODBC handle connection creation. In PyODBC, the
        token is passed outside the connection string in the `connect()` method. Since SQLAlchemy lazy-loads
        its connections, we need to set this as a separate argument to the `connect()` method. In SQLALchemy
        we do this by hooking into the `do_connect` event, which is fired when a connection is created.

        For further details, see:
        * <https://docs.sqlalchemy.org/en/20/dialects/mssql.html#connecting-to-databases-with-access-tokens>
        * <https://learn.microsoft.com/en-us/azure/azure-sql/database/azure-sql-python-quickstart
        """

        @event.listens_for(self.engine, "do_connect")
        def provide_token(_dialect, _conn_rec, cargs, cparams):
            # remove the "Trusted_Connection" parameter that SQLAlchemy adds
            cargs[0] = cargs[0].replace(";Trusted_Connection=Yes", "")

            # encode the token
            azure_token = self._auth_token.token
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

    def _add_embeddings_to_memory(self, *, embedding_data: list[EmbeddingDataEntry]) -> None:
        """
        Inserts embedding data into memory storage
        """
        self.insert_entries(entries=embedding_data)

    def _get_prompt_pieces_by_orchestrator(self, *, orchestrator_id: str) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptMemoryEntry Base objects that have the specified orchestrator ID.

        Args:
            orchestrator_id (str): The id of the orchestrator.
                Can be retrieved by calling orchestrator.get_identifier()["id"]

        Returns:
            list[PromptRequestPiece]: A list of PromptMemoryEntry Base objects matching the specified orchestrator ID.
        """
        try:
            sql_condition = text(
                "ISJSON(orchestrator_identifier) = 1 AND JSON_VALUE(orchestrator_identifier, '$.id') = :json_id"
            ).bindparams(json_id=str(orchestrator_id))
            entries = self.query_entries(PromptMemoryEntry, conditions=sql_condition)  # type: ignore
            result: list[PromptRequestPiece] = [entry.get_prompt_request_piece() for entry in entries]

            return result
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
            entries = self.query_entries(
                PromptMemoryEntry,
                conditions=PromptMemoryEntry.conversation_id == str(conversation_id),
            )  # type: ignore

            result: list[PromptRequestPiece] = [entry.get_prompt_request_piece() for entry in entries]
            return result

        except Exception as e:
            logger.exception(f"Failed to retrieve conversation_id {conversation_id} with error {e}")
            return []

    def add_request_pieces_to_memory(self, *, request_pieces: Sequence[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.

        """
        self.insert_entries(entries=[PromptMemoryEntry(entry=piece) for piece in request_pieces])

    def dispose_engine(self):
        """
        Dispose the engine and clean up resources.
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Engine disposed successfully.")

    def get_all_embeddings(self) -> list[EmbeddingDataEntry]:
        """
        Fetches all entries from the specified table and returns them as model instances.
        """
        result = self.query_entries(EmbeddingDataEntry)
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
            entries = self.query_entries(
                PromptMemoryEntry,
                conditions=PromptMemoryEntry.id.in_(prompt_ids),
            )
            result: list[PromptRequestPiece] = [entry.get_prompt_request_piece() for entry in entries]
            return result
        except Exception as e:
            logger.exception(
                f"Unexpected error: Failed to retrieve ConversationData with orchestrator {prompt_ids}. {e}"
            )
            return []

    def get_prompt_request_piece_by_memory_labels(
        self, *, memory_labels: dict[str, str] = {}
    ) -> list[PromptRequestPiece]:
        """
        Retrieves a list of PromptRequestPiece objects that have the specified memory labels.

        Args:
            memory_labels (dict[str, str]): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to an empty dictionary.

        Returns:
            list[PromptRequestPiece]: A list of PromptRequestPiece with the specified memory labels.
        """
        try:
            json_validation = "ISJSON(labels) = 1"
            json_conditions = " AND ".join([f"JSON_VALUE(labels, '$.{key}') = :{key}" for key in memory_labels])
            # Combine both conditions
            conditions = f"{json_validation} AND {json_conditions}"

            # Create SQL condition using SQLAlchemy's text() with bindparams
            # for safe parameter passing, preventing SQL injection
            sql_condition = text(conditions).bindparams(**{key: str(value) for key, value in memory_labels.items()})

            entries = self.query_entries(PromptMemoryEntry, conditions=sql_condition)
            result: list[PromptRequestPiece] = [entry.get_prompt_request_piece() for entry in entries]
            return result
        except Exception as e:
            logger.exception(
                f"Unexpected error: Failed to retrieve {PromptMemoryEntry.__tablename__} "
                f"with memory labels {memory_labels}. {e}"
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

    def insert_entry(self, entry: Base) -> None:  # type: ignore
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
    def insert_entries(self, *, entries: list[Base]) -> None:  # type: ignore
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

    def query_entries(
        self, model, *, conditions: Optional = None, distinct: bool = False  # type: ignore
    ) -> list[Base]:
        """
        Fetches data from the specified table model with optional conditions.

        Args:
            model: The SQLAlchemy model class corresponding to the table you want to query.
            conditions: SQLAlchemy filter conditions (Optional).
            distinct: Flag to return distinct rows (defaults to False).

        Returns:
            List of model instances representing the rows fetched from the table.
        """
        with closing(self.get_session()) as session:
            try:
                query = session.query(model)
                if conditions is not None:
                    query = query.filter(conditions)
                if distinct:
                    return query.distinct().all()
                return query.all()
            except SQLAlchemyError as e:
                logger.exception(f"Error fetching data from table {model.__tablename__}: {e}")
                return []

    def update_entries(self, *, entries: MutableSequence[Base], update_fields: dict) -> bool:  # type: ignore
        """
        Updates the given entries with the specified field values.

        Args:
            entries (list[Base]): A list of SQLAlchemy model instances to be updated.
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

    def print_schema(self):
        """Prints the schema of all tables in the Azure SQL database."""
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        for table_name in metadata.tables:
            table = metadata.tables[table_name]
            print(f"Schema for {table_name}:")
            for column in table.columns:
                print(f"  Column {column.name} ({column.type})")
