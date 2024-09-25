# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
import logging
import struct

from contextlib import closing
from typing import Optional, Sequence
import uuid
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken

from sqlalchemy import create_engine, event, text, MetaData, and_
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from pyrit.common import default_values
from pyrit.common.singleton import Singleton
from pyrit.memory.memory_models import Base, EmbeddingDataEntry, SeedPromptEntry, PromptMemoryEntry, ScoreEntry
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import AzureBlobStorageIO, SeedPrompt, SeedPromptGroup, PromptRequestPiece, PromptTemplate, Score

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
                conditions=PromptMemoryEntry.conversation_id == conversation_id,
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

    def query_entries(self, model, *, conditions: Optional = None, distinct: bool = False) -> list[Base]:  # type: ignore
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
                if distinct:
                    return query.distinct().all()
                return query.all()
            except SQLAlchemyError as e:
                logger.exception(f"Error fetching data from table {model.__tablename__}: {e}")

    def reset_database(self):
        """Drop and recreate existing tables"""
        # Drop all existing tables
        Base.metadata.drop_all(self.engine)
        # Recreate the tables
        Base.metadata.create_all(self.engine, checkfirst=True)

    def add_prompts_to_memory(self, *, prompts: list[SeedPrompt], added_by: Optional[str]=None) -> None:
        """
        Inserts a list of prompts into the memory storage.

        Args:
            prompts (list[SeedPrompt]): A list of prompts to insert.
            added_by (str): The user who added the prompts.
        """
        if added_by:
            for prompt in prompts:
                prompt.added_by = added_by
        if any([not prompt.added_by for prompt in prompts]):
            raise ValueError("One or more prompts did not have the 'added_by' attribute set.")
        
        for prompt in prompts:
            if prompt.date_added is None:
                prompt.date_added = datetime.now()

        self._insert_entries(entries=[SeedPromptEntry(entry=prompt) for prompt in prompts])
    
    def get_prompt_dataset_names(self) -> list[str]:
        """
        Returns a list of all prompt dataset names in the memory storage.
        """
        try:
            return self.query_entries(
                SeedPromptEntry.dataset_name,
                conditions=and_(SeedPromptEntry.dataset_name != None, SeedPromptEntry.dataset_name != ""),
                distinct=True,
            )  # type: ignore
        except Exception as e:
            logger.exception(f"Failed to retrieve dataset names with error {e}")
            return []
    
    def get_prompts(
        self,
        *,
        value: Optional[str] = None,
        dataset_name: Optional[str] = None,
        harm_categories: Optional[Sequence[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
        parameters: Optional[Sequence[str]] = None,
    ) -> list[SeedPrompt]:
        """
        Retrieves a list of prompts that have the specified dataset name.

        Args:
            value (str): The value to match by substring. If None, all values are returned.
            dataset_name (str): The dataset name to match. If None, all dataset names are considered.
            harm_categories (list[str]): A list of harm categories to filter by. If None, all harm categories are considered.
                Specifying multiple harm categories returns only prompts that are marked with all harm categories.
            added_by (str): The user who added the prompts.
            authors (list[str]): A list of authors to filter by.
                Note that this filters by substring, so a query for "Adam Jones" may not return results if the record
                is "A. Jones", "Jones, Adam", etc. If None, all authors are considered.
            groups (list[str]): A list of groups to filter by. If None, all groups are considered.
            source (str): The source to filter by. If None, all sources are considered.
            parameters (list[str]): A list of parameters to filter by. Specifying parameters effectively returns
                prompt templates instead of prompts.
                If None, only prompts without parameters are returned.

        Returns:
            list[SeedPrompt]: A list of prompts matching the criteria.
        """
        conditions = []
        if value:
            conditions.append(SeedPromptEntry.value.contains(value))
        if dataset_name:
            conditions.append(SeedPromptEntry.dataset_name == dataset_name)
        if harm_categories:
            for harm_category in harm_categories:
                conditions.append(SeedPromptEntry.harm_categories.contains(harm_category))
        if added_by:
            conditions.append(SeedPromptEntry.added_by == added_by)
        if authors:
            for author in authors:
                conditions.append(SeedPromptEntry.authors.contains(author))
        if groups:
            for group in groups:
                conditions.append(SeedPromptEntry.groups.contains(group))
        if source:
            conditions.append(SeedPromptEntry.source == source)
        if parameters:
            for parameter in parameters:
                conditions.append(SeedPromptEntry.parameters.contains(parameter))

        try:
            return self.query_entries(
                SeedPromptEntry,
                conditions=and_(*conditions),
            )  # type: ignore
        except Exception as e:
            logger.exception(f"Failed to retrieve prompts with dataset name {dataset_name} with error {e}")
            return []
    
    def get_prompt_templates(
        self,
        *,
        value: Optional[str] = None,
        dataset_name: Optional[str] = None,
        harm_categories: Optional[Sequence[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
        parameters: Optional[Sequence[str]] = None,
    ) -> list[PromptTemplate]:
        if not parameters:
            raise ValueError("Prompt templates must have parameters. Please specify at least one.")
        return [
            prompt.to_prompt_template() for prompt in self.get_prompts(
                value=value,
                dataset_name=dataset_name,
                harm_categories=harm_categories,
                added_by=added_by,
                authors=authors,
                groups=groups,
                source=source,
                parameters=parameters,
            )
        ]
    
    def get_prompt_groups(
        self,
        *,
        dataset_name: Optional[str] = None,
        data_types: Optional[Sequence[Sequence[str]]]
    ) -> list[SeedPromptGroup]:
        # TODO for this PR
        # join prompt tables as many times as the number of data types passed (which also implies the sequence number)
        # i.e., join on prompt group ID while matching the data type and sequence number
        # and optionally dataset_name and harm_categories
        raise NotImplementedError("Method not yet implemented.")

    def add_prompt_groups_to_memory(self, *, prompt_groups: list[SeedPromptGroup], added_by: Optional[str]=None) -> None:
        """
        Inserts a list of prompt groups into the memory storage.

        Args:
            prompt_groups (list[SeedPromptGroup]): A list of prompt groups to insert.
            added_by (str): The user who added the prompt groups.
        
        Raises:
            ValueError: If a prompt group does not have at least one prompt.
            ValueError: If prompt group IDs are inconsistent within the same prompt group.
        """
        # Validates the prompt group IDs and sets them if possible before leveraging the add_prompts_to_memory method.
        all_prompts = []
        for prompt_group in prompt_groups:
            if not prompt_group.prompts:
                raise ValueError("Prompt group must have at least one prompt.")
            # Determine the prompt group ID.
            # It should either be set uniformly or generated if not set.
            # Inconsistent prompt group IDs will raise an error.
            group_id_set = set([prompt.prompt_group_id for prompt in prompt_group.prompts])
            if len(group_id_set) > 1:
                raise ValueError(f"Inconsistent 'prompt_group_id' attribute between members of the same prompt group. Found {group_id_set}")
            prompt_group_id = group_id_set.pop() or str(uuid.uuid4())
            for prompt in prompt_group.prompts:
                prompt.prompt_group_id = prompt_group_id
            all_prompts.extend(prompt_group.prompts)
        self.add_prompts_to_memory(prompts=all_prompts, added_by=added_by)
    
    def delete_prompts(self, *, ids: list[str]) -> None:
        """
        Deletes prompts by id.
        """
        # TODO
        raise NotImplementedError("Method not yet implemented.")

    def print_schema(self):
        """Prints the schema of all tables in the Azure SQL database."""
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        for table_name in metadata.tables:
            table = metadata.tables[table_name]
            print(f"Schema for {table_name}:")
            for column in table.columns:
                print(f"  Column {column.name} ({column.type})")
