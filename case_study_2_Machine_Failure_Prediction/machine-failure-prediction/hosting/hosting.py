from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os, time

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
    time.sleep(5)  # let the space initialise before uploading

# Retry upload with exponential backoff for rate limiting (429)
max_retries = 5
for attempt in range(1, max_retries + 1):
    try:
        api.upload_folder(
            folder_path="machine-failure-prediction/deployment",
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo="",
        )
        print("Upload successful.")
        break
    except HfHubHTTPError as e:
        if "429" in str(e) and attempt < max_retries:
            wait = 2 ** attempt  # 2s, 4s, 8s, 16s, 32s
            print(f"Rate limited (429). Retrying in {wait}s... (attempt {attempt}/{max_retries})")
            time.sleep(wait)
        else:
            raise
