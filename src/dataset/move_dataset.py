from pathlib import Path
from tqdm import tqdm

source_dir = Path("data/raw_dataset")
destination_dir = Path("data/raw_dataset/html")
files_to_move = list(source_dir.glob("*"))

for file_path in tqdm(files_to_move):
    if file_path.is_file():
        file_path.rename(destination_dir / file_path.name)

print("All files moved successfully.")
