import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trak import TRAKer
from trak_utils import CausalLMModelOutput

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Adapted from: https://github.com/openai/simple-evals/blob/main/mmlu_eval.py
QUERY_TEMPLATE = """
Answer the following multiple choice question. Your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. 

{Question}

A) {A}
B) {B}
C) {C}
D) {D}

ANSWER: """.strip()

MODEL_NAME = "allenai/OLMo-1B"
RUN_NAME = 'mmlu'

TRAIN_SET_SIZE = 16
VAL_SET_SIZE = 16


class CausalLM(nn.Module):
    """
    Wrapper for OLMo model.
    """
    def __init__(self, model_name):
        super().__init__()
        print(f'Loading model {model_name}...')

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # self.model.eval().cuda()
        self.model.train().cuda()

    # /srv/nlprx-lab/share6/dheineman3/miniconda3/envs/trak/lib/python3.10/site-packages/hf_olmo/modeling_olmo.py
    def forward(self, input_ids, attention_mask):
        generation = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1, 
            do_sample=False,
            top_p=1,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True
        )
        # output = generation.sequences
        
        # Get token scores
        scores = generation.scores[0]

        return scores


def get_dataset(split, inds=None, limit=None):
    # https://huggingface.co/datasets/cais/mmlu
    raw_datasets = load_dataset(
        "cais/mmlu", 
        "all"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side='left',
        trust_remote_code=True
    )

    label_list = ['A', 'B', 'C', 'D']
    label_toks = {i: tokenizer(label)['input_ids'][0] for i, label in enumerate(label_list)}
    # label_list = [0, 1, 2, 3]
    # label_to_id = {v: i for i, v in enumerate(label_list)}
    
    print(f'Tokenized labels: {label_list} -> {label_toks}')

    def preprocess_function(examples):
        question, subject, choices, answer = examples['question'], examples['subject'], examples['choices'], examples['answer']

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
        desc="Running tokenizer on dataset",
    )

    # 'input_ids', 'attention_mask', 'label'
    ds = ds.remove_columns(['question', 'subject', 'choices', 'answer'])
    print(ds)
    
    return ds


def init_loaders(batch_size=4):
    # Corresponds to the HF dataset split
    # auxilary_train, dev, test, validation
    ds_train = get_dataset('test', limit=TRAIN_SET_SIZE) 
    ds_val   = get_dataset('validation', limit=VAL_SET_SIZE)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    loader_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

    return loader_train, loader_val


def process_batch(batch):
    return batch['input_ids'], batch['attention_mask'], batch['labels']


def main(ckpt, out, device='cuda'):
    loader_train, loader_val = init_loaders()

    model = CausalLM(MODEL_NAME)

    out = os.path.join(out, RUN_NAME)

    traker = TRAKer(
        model=model,
        task=CausalLMModelOutput,
        train_set_size=TRAIN_SET_SIZE,
        save_dir=out,
        device=device,
        proj_dim=1024
    )

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in tqdm(loader_train, desc='Featurizing..'):
        batch = process_batch(batch)
        batch = [x.to(device) for x in batch]
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    traker.finalize_features()

    traker.start_scoring_checkpoint(
        exp_name=RUN_NAME,
        checkpoint=model.state_dict(),
        model_id=0,
        num_targets=VAL_SET_SIZE
    )

    for batch in tqdm(loader_val, desc='Scoring..'):
        batch = process_batch(batch)
        batch = [x.to(device) for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name=RUN_NAME)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='model checkpoint', default=None)
    parser.add_argument('--out', type=str, help='dir to save TRAK scores and metadata to', default='../results')
    args = parser.parse_args()
    main(args.ckpt, args.out)