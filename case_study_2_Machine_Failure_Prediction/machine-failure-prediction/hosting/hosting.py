from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id   = "partha90/Machine-Failure-Prediction"
repo_type = "space"
token     = os.getenv("HF_TOKEN")

api = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="streamlit",
        private=False,
        token=token,
    )
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="machine-failure-prediction/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)
