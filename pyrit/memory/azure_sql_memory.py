# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import struct
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import MutableSequence, Optional, Sequence, TypeVar, Union

from azure.core.credentials import AccessToken
from azure.identity import DefaultAzureCredential
from sqlalchemy import MetaData, create_engine, event, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload, sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.common import default_values
from pyrit.common.singleton import Singleton
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.memory.memory_models import Base, EmbeddingDataEntry, PromptMemoryEntry
from pyrit.models import AzureBlobStorageIO, PromptRequestPiece

logger = logging.getLogger(__name__)

Model = TypeVar("Model")


class AzureSQLMemory(MemoryInterface, metaclass=Singleton):
    """
    A class to manage conversation memory using Azure SQL Server as the backend database. It leverages SQLAlchemy Base
    models for creating tables and provides CRUD operations to interact with the tables.

    This class encapsulates the setup of the database connection, table creation based on SQLAlchemy models,
    and session management to perform database operations.
    """

    # Azure SQL configuration
    SQL_COPT_SS_ACCESS_TOKEN = 1256  # Connection option for access tokens, as defined in msodbcsql.h
    TOKEN_URL = "https://database.windows.net/.default"  # The token URL for any Azure SQL database
    AZURE_SQL_DB_CONNECTION_STRING = "AZURE_SQL_DB_CONNECTION_STRING"

    # Azure Storage Account Container datasets and results environment variables
    AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL: str = "AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL"
    AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN: str = "AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN"

    def __init__(
        self,
        *,
        connection_string: Optional[str] = None,
        results_container_url: Optional[str] = None,
        results_sas_token: Optional[str] = None,
        verbose: bool = False,
    ):
        self._connection_string = default_values.get_required_value(
            env_var_name=self.AZURE_SQL_DB_CONNECTION_STRING, passed_value=connection_string
        )

        self._results_container_url: str = default_values.get_required_value(
            env_var_name=self.AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL, passed_value=results_container_url
        )

        self._results_container_sas_token: Optional[str] = self._resolve_sas_token(
            self.AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN, results_sas_token
        )

        self._auth_token: Optional[AccessToken] = None
        self._auth_token_expiry: Optional[int] = None

        self.results_path = self._results_container_url

        self.engine = self._create_engine(has_echo=verbose)

        # Generate the initial auth token
        self._create_auth_token()
        # Enable token-based authorization
        self._enable_azure_authorization()

        self.SessionFactory = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()

        super(AzureSQLMemory, self).__init__()

    @staticmethod
    def _resolve_sas_token(env_var_name: str, passed_value: Optional[str]) -> Optional[str]:
        """
        Resolve the SAS token value, allowing a fallback to None for delegation SAS.

        Args:
            env_var_name (str): The environment variable name to look up.
            passed_value (Optional[str]): A passed-in value for the SAS token.

        Returns:
            Optional[str]: Resolved SAS token or None if not provided.
        """
        try:
            return default_values.get_required_value(env_var_name=env_var_name, passed_value=passed_value)
        except ValueError:
            return None

    def _init_storage_io(self):
        # Handle for Azure Blob Storage when using Azure SQL memory.
        self.results_storage_io = AzureBlobStorageIO(
            container_url=self._results_container_url, sas_token=self._results_container_sas_token
        )

    def _create_auth_token(self) -> None:
        """
        Creates an Azure Entra ID access token.
        Stores the token and its expiry time.
        """
        azure_credentials = DefaultAzureCredential()
        token: AccessToken = azure_credentials.get_token(self.TOKEN_URL)
        self._auth_token = token
        self._auth_token_expiry = token.expires_on

    def _refresh_token_if_needed(self) -> None:
        """
        Refresh the access token if it is close to expiry (within 5 minutes).
        """
        if datetime.now(timezone.utc) >= datetime.fromtimestamp(self._auth_token_expiry, tz=timezone.utc) - timedelta(
            minutes=5
        ):
            logger.info("Refreshing Microsoft Entra ID access token...")
            self._create_auth_token()

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
            # Refresh token if it's close to expiry
            self._refresh_token_if_needed()

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

    def _add_embeddings_to_memory(self, *, embedding_data: Sequence[EmbeddingDataEntry]) -> None:
        """
        Inserts embedding data into memory storage
        """
        self._insert_entries(entries=embedding_data)

    def _get_prompt_pieces_memory_label_conditions(self, *, memory_labels: dict[str, str]):
        json_validation = "ISJSON(labels) = 1"
        json_conditions = " AND ".join([f"JSON_VALUE(labels, '$.{key}') = :{key}" for key in memory_labels])
        # Combine both conditions
        conditions = f"{json_validation} AND {json_conditions}"

        # Create SQL condition using SQLAlchemy's text() with bindparams
        # for safe parameter passing, preventing SQL injection
        return text(conditions).bindparams(**{key: str(value) for key, value in memory_labels.items()})

    def _get_prompt_pieces_orchestrator_conditions(self, *, orchestrator_id: str):
        return text(
            "ISJSON(orchestrator_identifier) = 1 AND JSON_VALUE(orchestrator_identifier, '$.id') = :json_id"
        ).bindparams(json_id=str(orchestrator_id))

    def _get_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]):
        json_validation = "ISJSON(prompt_metadata) = 1"
        json_conditions = " AND ".join([f"JSON_VALUE(prompt_metadata, '$.{key}') = :{key}" for key in prompt_metadata])
        # Combine both conditions
        conditions = f"{json_validation} AND {json_conditions}"

        # Create SQL condition using SQLAlchemy's text() with bindparams
        # for safe parameter passing, preventing SQL injection
        return text(conditions).bindparams(**{key: str(value) for key, value in prompt_metadata.items()})

    def _get_prompt_pieces_prompt_metadata_conditions(self, *, prompt_metadata: dict[str, Union[str, int]]):
        return self._get_metadata_conditions(prompt_metadata=prompt_metadata)

    def _get_seed_prompts_metadata_conditions(self, *, metadata: dict[str, Union[str, int]]):
        return self._get_metadata_conditions(prompt_metadata=metadata)

    def add_request_pieces_to_memory(self, *, request_pieces: Sequence[PromptRequestPiece]) -> None:
        """
        Inserts a list of prompt request pieces into the memory storage.

        """
        self._insert_entries(entries=[PromptMemoryEntry(entry=piece) for piece in request_pieces])

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
        self, Model, *, conditions: Optional = None, distinct: bool = False, join_scores: bool = False  # type: ignore
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

    def print_schema(self):
        """Prints the schema of all tables in the Azure SQL database."""
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        for table_name in metadata.tables:
            table = metadata.tables[table_name]
            print(f"Schema for {table_name}:")
            for column in table.columns:
                print(f"  Column {column.name} ({column.type})")
