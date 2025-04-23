import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login

def upload_to_huggingface(file_path, repo_id, repo_type="dataset"):
    """
    Upload a local file to Hugging Face Hub.
    
    Args:
        file_path: Path to the local file to upload
        repo_id: The Hugging Face repository ID (e.g., 'username/repo-name')
        repo_type: Repository type ('dataset' or 'model')
    """
    # Load environment variables
    load_dotenv()
    
    # Get token from environment variable
    token = os.getenv("HF_WRITE_TOKEN")
    if not token:
        raise ValueError("HF_WRITE_TOKEN environment variable is required. Please set it first.")
    
    # Login to Hugging Face
    os.environ["HF_TOKEN"] = token
    login(token=token)
    print(f"Logged in to Hugging Face Hub")
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repository {repo_id} is ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload the file
    file_name = os.path.basename(file_path)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    
    print(f"Successfully uploaded {file_path} to {repo_id}")
    print(f"View it at: https://huggingface.co/{repo_type}s/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload a local file to Hugging Face Hub')
    parser.add_argument('file_path', help='Path to the local file to upload')
    parser.add_argument('repo_id', help='The Hugging Face repository ID (e.g., "username/repo-name")')
    parser.add_argument('--repo-type', default='dataset', choices=['dataset', 'model'], 
                        help='Repository type (default: dataset)')
    
    args = parser.parse_args()
    
    upload_to_huggingface(args.file_path, args.repo_id, args.repo_type) 