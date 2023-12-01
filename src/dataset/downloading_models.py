from pathlib import Path

from huggingface_hub import snapshot_download
import subprocess
import shutil

# IMPORTANT: Run this script from rag dir

# Author: repo name
repo_model_ids = {
    "codellama": "CodeLlama-34b-Instruct-hf",
    "Yhyu13": "oasst-rlhf-2-llama-30b-7k-steps-hf",
}

# Author: (repo_name, specific file)
gguf_model_ids = {
    # "TheBloke": ("Mistral-7b-OpenOrca-GGUF", "mistral-7b-openorca.Q4_0.gguf")
}


def delete_dir(path: Path):
    try:
        shutil.rmtree(path)
        print(f"Directory '{path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {e}")


def download_hf_repos(repo_model_ids: dict):
    for model_author, model_name in list(repo_model_ids.items()):
        snapshot_download(
            repo_id=f"{model_author}/{model_name}",
            local_dir=f"models/{model_author}-{model_name}",
            local_dir_use_symlinks=False,
            revision="main",
        )
        command = f"python src/dataset/convert-hf-to-gguf.py models/{model_author}-{model_name} --outfile models/{model_author}-{model_name}.gguf --outtype q8_0"

        subprocess.call(command, shell=True)
        delete_dir(Path(f"models/{model_author}-{model_name}"))


def download_gguf_models(gguf_model_ids: dict):
    for model_author, model_name_tuple in list(gguf_model_ids.items()):
        repo_name, model_name = model_name_tuple
        folder_name = Path(f"models/{model_author}-{model_name}")

        snapshot_download(
            repo_id=f"{model_author}/{repo_name}",
            local_dir=folder_name,
            allow_patterns=model_name,
            local_dir_use_symlinks=False,
            revision="main",
        )

        parent_dir = folder_name.parent
        new_file_path = parent_dir / model_name
        Path(folder_name / model_name).rename(new_file_path)
        delete_dir(folder_name)


if __name__ == "__main__":
    download_hf_repos(repo_model_ids)
    download_gguf_models(gguf_model_ids)
