import os
import subprocess
from pyrit.common import default_values
import time
from huggingface_hub import HfApi  # Hugging Face API for model size


# Load environment variables
default_values.load_default_env()


# Set environment variables for huggingface-cli
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable parallel downloads


def login_to_huggingface():
    """Logs into Hugging Face using the token provided as an environment variable."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set. Please set it before running this function.")

    command = [
        "huggingface-cli", 
        "login", 
        "--token", 
        token, 
        "--add-to-git-credential"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("Successfully logged into Hugging Face.")
    except subprocess.CalledProcessError as e:
        print(f"Error logging in to Hugging Face: {e}")
        print(e.stderr)
        raise


def get_available_files(model_id: str):
    """Fetches available files for a model from the Hugging Face repository."""
    api = HfApi()
    try:
        model_info = api.model_info(model_id)
        available_files = [file.rfilename for file in model_info.siblings]
        return available_files
    except Exception as e:
        print(f"Error fetching model files for {model_id}: {e}")
        return []
    

def download_model_with_cli(model_id: str):
    """Downloads a Hugging Face model using huggingface-cli with parallel transfer enabled.

    Args:
        model_id: The model ID from Hugging Face.

    Raises:
        subprocess.CalledProcessError: If the huggingface-cli command fails.
    """
    print("Please make sure you are logged in to Hugging Face CLI. Running login process...")
    login_to_huggingface()  # Log in before downloading

    # Command to run huggingface-cli
    command = [
        "huggingface-cli",
        "download",
        model_id,
        "--repo-type", "model"  # Ensure we specify the repository type
    ]

    start_time = time.time()

    try:
        # Run the huggingface-cli command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"Model {model_id} downloaded successfully to the cache.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model {model_id}: {e}")
        print(e.stderr)
        raise

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Download completed in {elapsed_time:.2f} seconds.")


def download_specific_files_with_cli(model_id: str, file_patterns: list):
    """Downloads specific files from a Hugging Face model repository using huggingface-cli.

    Args:
        model_id: The model ID from Hugging Face.
        file_patterns: A list of file patterns to download.

    Raises:
        subprocess.CalledProcessError: If the huggingface-cli command fails.
    """
    print("Please make sure you are logged in to Hugging Face CLI. Running login process...")
    login_to_huggingface()

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{model_id.replace('/', '--')}")
    available_files = get_available_files(model_id)
    files_to_download = [file for file in available_files if any(pattern in file for pattern in file_patterns)]
    
    if not files_to_download:
        print(f"No files matched the patterns provided for model {model_id}.")
        return

    print("The following files will be downloaded:")
    for file in files_to_download:
        print(f"  - {file}")

    command = [
        "huggingface-cli",
        "download",
        model_id,
        "--repo-type", "model",
        "--local-dir", cache_dir
    ]

    for pattern in file_patterns:
        command.extend(["--include", pattern])

    start_time = time.time()

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        downloaded_files = [line.split()[-1] for line in result.stdout.splitlines() if "Downloading" in line]
        
        print("Downloaded files:")
        for file in downloaded_files:
            print(f"  - {file}")
        print(f"Specified files for {model_id} downloaded successfully to the cache.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading specific files for model {model_id}: {e}")
        print(e.stderr)
        raise

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Download of specific files completed in {elapsed_time:.2f} seconds.")


# Main function to test the functionality
def main():
    # Test login function
    try:
        print("Testing Hugging Face CLI login...")
        login_to_huggingface()
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")

    # Test downloading specific files
    model_id = "microsoft/Phi-3-mini-4k-instruct"  # Replace with any model you want to test
    file_patterns = [
        "model-00001-of-00002.safetensors",  # Model weights part 1
        "model-00002-of-00002.safetensors",  # Model weights part 2
        "config.json",  # Model configuration
        "tokenizer.json",  # Tokenizer configuration
        "tokenizer.model",  # Tokenizer model file (e.g., SentencePiece)
        "special_tokens_map.json",  # Special tokens mapping
        "generation_config.json",  # Generation configuration (if applicable)
        "vocab.json",  # Vocabulary file for tokenizers like BERT
        "merges.txt",  # Merge rules for BPE tokenizers like GPT-2
    ]
    try:
        print(f"\nTesting specific file download for model ID: {model_id}...")
        download_specific_files_with_cli(model_id, file_patterns)
        print(f"Specified files for {model_id} downloaded successfully.")
    except Exception as e:
        print(f"Specific file download failed: {e}")


if __name__ == "__main__":
    main()
