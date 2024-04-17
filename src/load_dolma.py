import os
from datasets import load_dataset

os.environ["DATA_DIR"] = "<path_to_your_data_directory>"
dataset = load_dataset("allenai/dolma", split="train")
