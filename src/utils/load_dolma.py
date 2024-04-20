import os
from datasets import load_dataset

os.environ["DATA_DIR"] = "/nethome/dheineman3/nlprx/trak/data/data"

dataset = load_dataset(
    "allenai/dolma", 
    split="train", 
    name="v1_6-sample",
    trust_remote_code=True
)

print(dataset)