import os
import subprocess
import requests
import tarfile
import zipfile

def check_command(command):
    """Check if a command is available on the system."""
    result = subprocess.run(['which', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"'{command}' command not found, please install it to prepare data")
        exit(1)

def download_file(url, dest):
    """Download a file from a URL to a destination."""
    if not os.path.exists(dest):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"File {dest} already exists. Skipping download.")

def extract_tar_gz(file_path, dest_dir):
    """Extract a .tar.gz file to a destination directory."""
    # if not os.path.exists(dest_dir):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dest_dir)
    # else:
    #     print(f"Directory {dest_dir} already exists. Skipping extraction.")

def extract_zip(file_path, dest_dir):
    """Extract a .zip file to a destination directory."""
    if not os.path.exists(dest_dir):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    else:
        print(f"Directory {dest_dir} already exists. Skipping extraction.")

def main():
    # Check if 'unzip' command is available
    check_command('unzip')

    try:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download spaCy model: {e}")

    # Create sample problematic example data
    os.makedirs('sbf', exist_ok=True)
    download_file('https://maartensap.com/social-bias-frames/SBIC.v2.tgz', 'sbf/SBIC.v2.tgz')
    extract_tar_gz('sbf/SBIC.v2.tgz', 'sbf/')
    if not os.path.exists('sample_utterances.json'):
        subprocess.run(['python', 'gen_sample_utterances.py'], check=True)
    else:
        print("Sample utterances already generated. Skipping.")

    # Download data which will be sampled for few-shot
    os.makedirs('entailmentbank', exist_ok=True)
    download_file('https://huggingface.co/datasets/ariesutiono/entailment-bank-v3/raw/main/task1_train.jsonl', 'entailmentbank/task1_train.jsonl')
    download_file('https://github.com/nyu-mll/nope/raw/main/annotated_corpus/nope-v1.zip', 'nope-v1.zip')
    extract_zip('nope-v1.zip', 'nope')

    # if not os.path.exists('data_generation_imppres'):
    #     subprocess.run(['git', 'clone', '-b', 'imppres', '--single-branch', 'git@github.com:alexwarstadt/data_generation.git', 'data_generation_imppres'], check=True)
    # else:
    #     print("Data generation repository already cloned. Skipping.")

    # Change directory to one level up before running exemplars.py
    os.chdir('..')

    # Process data for few-shot
    if not os.path.exists('exemplars_processed.json'):
        subprocess.run(['python', 'exemplars.py'], check=True)
    else:
        print("Exemplars already processed. Skipping.")

if __name__ == '__main__':
    main()