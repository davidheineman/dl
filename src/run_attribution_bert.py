import json, os

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from tqdm.auto import tqdm

from rake_nltk import Rake
import nltk

MODEL_PATH = "../results/bert-attribution-suhani-small"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join('..', 'data', 'alpaca')


class ScoresDataset(Dataset):
    def __init__(self, tokenizer, train_path, test_path, limit=None):
        self.tokenizer = tokenizer

        with open(train_path, 'r') as f:
            train_data = [json.loads(l) for l in f if l.strip()]

        with open(test_path, 'r') as f:
            test_data = [json.loads(l) for l in f if l.strip()]

        self.input_text  = [entry['instruction'] for entry in train_data]
        self.labels_text = [entry['label'] for entry in train_data]
        self.test_text   = [entry['input_text'] for entry in test_data]

        print(f'Loaded data: \n  Train data of {len(train_data)} documents\n  Test data of {len(test_data)} examples\n')

        count = 0
        self.pairs_list = []
        for train_idx, _ in enumerate(train_data):
            labels = self.labels_text[train_idx]
            for token_idx, _ in enumerate(labels):
                if token_idx > 1: continue
                for test_idx, _ in enumerate(test_data):
                    if test_idx > 1: continue
                    self.pairs_list += [(
                        self.input_text[train_idx],
                        self.labels_text[train_idx][token_idx],
                        self.test_text[test_idx]
                    )]

                    count += 1
                    if limit and count >= limit: return

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        input1, input2, input3 = self.pairs_list[idx]
        total_input = input1 + " " + input2 + " " + input3

        tokenized = self.tokenizer(total_input, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        return tokenized
    

def nltk_setup():
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')
    nltk.download('punkt')
    

def download_alpaca():
    nltk_setup()

    # Download dataset from HF
    ds = load_dataset("tatsu-lab/alpaca")
    ds['train'].to_json(os.path.join(DATA_PATH, 'train_data.json'))
    
    # Apply NLTK to dataset
    with open(os.path.join(DATA_PATH, 'train_data.json'), 'r') as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    df = pd.DataFrame(train_data, columns=['instruction', 'input', 'output', 'text'])
    df.index = range(len(df))
    r = Rake()
    def extract_keywords(instruction):
        r.extract_keywords_from_text(instruction)
        return r.get_ranked_phrases()
    df['label'] = df['instruction'].apply(extract_keywords)
    df.to_json(os.path.join(DATA_PATH, 'train_data.json'), orient='records', lines=True)


def download_mmlu(split, limit=None):
    from utils.dataloader import MMLU_TEMPLATE
    
    raw_datasets = load_dataset("cais/mmlu", "all")
    query_template = MMLU_TEMPLATE

    def preprocess_function(examples):
        question, _, choices, answer = examples['question'], examples['subject'], examples['choices'], examples['answer']

        input_text = [
            query_template.format(
                Question=q, 
                A=choices[i][0], B=choices[i][1], C=choices[i][2], D=choices[i][3]
            ) for i, q in enumerate(question)
        ]
        
        return {"input_text": input_text}

    ds = raw_datasets[split]
    if limit:
        ds = ds.select(range(limit))

    ds = ds.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
        remove_columns=[col for col in raw_datasets[split].column_names if col not in ["input_ids", "labels", "attention_mask"]],
        desc="Pre-processing dataset",
    )

    ds.to_json(os.path.join(DATA_PATH, 'test_data.json'))


def save_entries_at_indices(indices, jsonl_file, output_file):
    with open(jsonl_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for idx, line in enumerate(f_in):
                if idx in indices: f_out.write(line)


def main():
    download_alpaca()
    download_mmlu('test')

    model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    total_dataset = ScoresDataset(
        tokenizer,
        os.path.join(DATA_PATH, f'train_data.json'),
        os.path.join(DATA_PATH, f'test_data.json'),
        limit=None
    )

    dataloader = DataLoader(
        total_dataset, 
        batch_size=512, 
        shuffle=True, 
        generator=torch.Generator(device='cpu')
    )

    model.eval()

    predictions = []
    with torch.no_grad():
        iterate = tqdm(dataloader, desc='Running BERT attribution', leave=True)
        for batch in iterate:
            input = {k: v.squeeze().to(DEVICE) for k, v in batch.items()}

            if (len(input['input_ids'].shape) != 2):
                input = {k: v.reshape(1, input['input_ids'].shape[0]) for k, v in input.items()}

            output = model(**input)
            cur_pred = output.logits.squeeze().to(DEVICE)
            predictions.append(cur_pred)

    predictions = torch.cat(predictions, dim=0)
    torch.save(predictions, '../results/alpaca-attribution/prediction_scores.pt')
    
    print(f'Calculated attributions: {predictions.shape}')

    # Get top 5% of scoring outputs
    top_k_percent = 5
    n_examples = int(predictions.numel() * (top_k_percent / 100))
    _, top_entries = torch.topk(predictions.view(-1), n_examples)
    save_entries_at_indices(
        top_entries.tolist(),
        os.path.join(DATA_PATH, f'stanford_alpaca_data.jsonl'), 
        os.path.join(DATA_PATH, f'alpaca_top_entries.jsonl')
    )
    
    # Get random scoring outputs
    max_index = predictions.numel()
    random_entries = torch.randint(0, max_index, (n_examples,))
    save_entries_at_indices(
        random_entries.tolist(), 
        os.path.join(DATA_PATH, f'stanford_alpaca_data.jsonl'), 
        os.path.join(DATA_PATH, f'alpaca_random_entries.jsonl')
    )
    


if __name__ == '__main__':
    main()