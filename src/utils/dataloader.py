from functools import partial
import torch
from datasets import load_dataset

from utils.load_tulu import encode_with_messages_format, pad


# Adapted from: https://github.com/openai/simple-evals/blob/main/mmlu_eval.py
QUERY_TEMPLATE = """
Answer the following multiple choice question. Your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. 

{Question}

A) {A}
B) {B}
C) {C}
D) {D}

ANSWER: """.strip()
MMLU_LABELS = ['A', 'B', 'C', 'D']


def get_dataset(split, tokenizer, inds=None, limit=None):
    # https://huggingface.co/datasets/cais/mmlu
    raw_datasets = load_dataset(
        "cais/mmlu", 
        "all"
    )

    label_list = MMLU_LABELS
    label_toks = {i: tokenizer(label)['input_ids'][0] for i, label in enumerate(label_list)}
    # label_list = [0, 1, 2, 3]
    
    print(f'Tokenized labels: {label_list} -> {label_toks}')

    def preprocess_function(examples):
        question, _, choices, answer = examples['question'], examples['subject'], examples['choices'], examples['answer']

        input_text = [
            QUERY_TEMPLATE.format(
                Question=q, 
                A=choices[i][0], B=choices[i][1], C=choices[i][2], D=choices[i][3]
            ) for i, q in enumerate(question)
        ]
        
        input_toks = tokenizer(
            input_text, 
            return_tensors='pt', 
            return_token_type_ids=False,
            padding=True,
            # padding = "max_length",
            # max_seq_length = 128,
        )
        input_toks["labels"] = torch.Tensor([label_toks[a] for a in answer])

        assert all(isinstance(e, torch.Tensor) for e in input_toks.values())
        
        return input_toks

    ds = raw_datasets[split]
    if limit:
        ds = ds.select(range(limit))

    ds = ds.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
        remove_columns=[col for col in raw_datasets[split].column_names if col not in ["input_ids", "labels", "attention_mask"]],
        desc="Running tokenizer on dataset",
    )
    
    print(ds)

    for e in ds:
        # print(e['input_ids'].shape, e['labels'].shape, e['attention_mask'].shape)
        print(e)
    
    return ds


def load_tulu_dataset(split, tokenizer, train_file, max_seq_length=2048, overwrite_cache=False, preprocessing_num_workers=1, add_bos=True, limit=None):
    """
    Loads and tokenizes a Tulu format dataset
    """
    data_files = {}
    dataset_args = {}
    if train_file is not None:
        data_files["train"] = train_file
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    if "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            add_bos=add_bos,
        )
    else:
        raise NotImplementedError()
        
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        remove_columns=[col for col in raw_datasets["train"].column_names if col not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    ds = lm_datasets[split]
    if limit:
        ds = ds.select(range(limit))

    # print(ds)
    for e in ds:
        print(e)
        # print(e['input_ids'])
        # print(e['labels'])
        # print(e['attention_mask'])

        # print(e['input_ids'].shape, e['labels'].shape, e['attention_mask'].shape)

        # features = pad(tokenizer, [e], padding='max_length', max_length=2048)

    return ds
