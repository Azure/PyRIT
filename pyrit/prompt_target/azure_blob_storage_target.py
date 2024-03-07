# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
import os
import sys
import tempfile
import uuid

from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptTarget


class AzureBlobStorageTarget(PromptTarget):
    """
    The StorageAccountTarget takes prompts, saves the prompts to a file, and stores them as a blob in a given storage account container.

    container_url: TODO ADD PARAMETER DESCRIPTION
    memory: TODO ADD PARAMETER DESCRIPTION
    """

    def __init__(
            self,
            *,
            container_url: str = None,
            sas_token: str = None,
            memory: MemoryInterface = None,
        ) -> None:
        
        load_dotenv()
        if container_url == None:
            try:
                container_url = os.getenv('AZURE_STORAGE_ACCOUNT_CONTAINER_URL')
                self.container_url = container_url
            except:
                raise Exception("Make sure your .env file is populated with a value for 'AZURE_STORAGE_ACCOUNT_CONTAINER_URL' for this storage account container")
            
        if sas_token == None:
            try:
                sas_token = os.getenv('AZURE_STORAGE_ACCOUNT_SAS_TOKEN')
            except:
                raise Exception("Make sure your .env file is populated with a value for 'AZURE_STORAGE_ACCOUNT_SAS_TOKEN' for this storage account container")
            
        self.container_client = ContainerClient.from_container_url(
            container_url=container_url,
            credential=sas_token)
        
        self.created_blob_url_list = []

        super().__init__(memory)

        # container URL / this is created by the operator and connection string + sas key given for this
        # there can be containers that have anonymous access...how to factor this in? (by default it is private to the account owner) (can enable 'blob' or 'container' for anonymous list/read for the storage account scope)

    """
    List all blob names that are in the container
    TODO: Double check if this is a necessary function
    TODO: Alternatively can return the created_blob_url_list saved, with the full blob URL
    TODO: Alternatively can use this, but prefix the blob names with container URLs depending on use case
    """
    def list_blob_names(self) -> list[str]:
        return self.container_client.list_blob_names()

    def send_prompt(
            self,
            normalized_prompt: str,
            conversation_id: str,
            normalizer_id: str
        ) -> str:

        # Create temporary file to upload to Storage Account Container URL
        local_file_name =  str(uuid.uuid4()) + ".txt"

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_file_path = os.path.join(temp_dir, local_file_name)

            file = open(file=upload_file_path, mode='w')
            file.write(normalized_prompt)
            file.close()

            print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

            with open(file=upload_file_path, mode='rb') as file_data:
                self.container_client.upload_blob(
                    name=local_file_name,
                    data=file_data,
                    length=sys.getsizeof(upload_file_path))
        
        # Track created blob URL
        self.created_blob_url_list.append(self.container_url + "/ "+ local_file_name)

        # Add prompt to memory
        """
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )
        """
