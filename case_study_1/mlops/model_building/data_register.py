from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "partha90/bank-customer-churn"
repo_type = "dataset"
token = os.getenv("HF_TOKEN")

api = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Dataset repo '{repo_id}' created.")
except HfHubHTTPError as e:
    print(f"HTTP error while checking repo (check HF_TOKEN permissions): {e}")
    raise

api.upload_folder(
    folder_path="mlops/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
