import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="khmer-synth.h5",
    path_in_repo="khmer-synth.h5",
    repo_id="ishinvin/khmer-synth",
    repo_type="dataset",
)
