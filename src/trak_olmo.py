import os, datetime, json
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trak import TRAKer
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from hf_olmo import OLMoTokenizerFast

from utils.trak_output import CausalLMModelOutput
from utils.dataloader import load_tulu_dataset, load_hf_dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# MODEL_NAME = 'allenai/OLMo-1B'
MODEL_NAME = 'davidheineman/OLMo-1B-Instruct'
TRAIN_FILE = '../data/lima.jsonl'

RUN_NAME = 'lima' + datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")

TRAIN_SET_SIZE  = 65536  # LIMA is ~150K tokens (166048)
MAX_SEQ_LEN     = 256
VAL_SET_SIZE    = 13000   # MMLU test set is 14K examples
BATCH_SIZE      = 1


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

        self.model.train().cuda()

    def forward(self, input_ids, attention_mask):
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        scores = output.logits
        scores = scores[..., -1, :] # Get the last token (-1)

        return scores


def init_loaders(tokenizer, batch_size):
    ds_train = load_tulu_dataset(
        'train', tokenizer, TRAIN_FILE, limit=TRAIN_SET_SIZE, 
        overwrite_cache=True, max_seq_length=MAX_SEQ_LEN
    )
    ds_val = load_hf_dataset('mmlu', 'test', tokenizer, limit=VAL_SET_SIZE)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    loader_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

    return loader_train, loader_val


def process_batch(batch):
    return batch['input_ids'], batch['attention_mask'], batch['labels']


def list_to_tensor(lst):
    """
    Maps a 3D python list to a 3D tenor, broadcasting '-inf' to the rest of the tensor
    """
    max_inner_length = max(len(sublst) for sublst in lst)
    max_middle_length = max(len(subsublst) for sublst in lst for subsublst in sublst)
    padded_lst = []
    for sublst in lst:
        padded_sublst = []
        for subsublst in sublst:
            padded_subsublst = subsublst + [-float('inf')] * (max_middle_length - len(subsublst))
            padded_sublst.append(padded_subsublst)
        padded_sublst += [[-float('inf')] * max_middle_length] * (max_inner_length - len(sublst))
        padded_lst.append(padded_sublst)
    return torch.tensor(padded_lst)


def save_outputs(out_path, scores, loader_train, loader_val, tokenizer):
    """
    Save both datasets as a JSON, with all metadata
    """
    out_text = {}
    out_score = {}
    for i, batch in tqdm(enumerate(loader_train), desc='Saving scores over training data'):
        input_ids, labels, e_ids, all_input_ids = batch['input_ids'], batch['labels'], batch['example_id'], batch['full_input_ids']
        for j in range(input_ids.shape[0]):
            id_ = i + j
            input_toks, label_toks, e_id, all_input_toks = input_ids[j], labels[j], int(e_ids[j]), all_input_ids[j]
            if e_id not in out_text:
                out_text[e_id] = {
                    'input_text': tokenizer.batch_decode([all_input_toks], skip_special_tokens=True)[0],
                    'labels_text': []
                }
            out_text[e_id]['labels_text'] += tokenizer.batch_decode([label_toks], skip_special_tokens=True)
            
            if e_id not in out_score: out_score[e_id] = []
            out_score[e_id] += [scores[id_, :].tolist()]
    out_text  = [out_text[i] for i in sorted(list(out_text.keys()))] # Convert output to list
    out_score = [out_score[i] for i in sorted(list(out_score.keys()))]
    out_score = list_to_tensor(out_score)

    print(out_score.shape)

    with open(os.path.join(out_path, 'trak_train.json'), "w") as f:
        json.dump(out_text, f) # indent=4
    torch.save(out_score, os.path.join(out_path, 'attribution_scores.pt'))

    output = []
    for i, batch in tqdm(enumerate(loader_val), desc='Saving test data'):
        input_ids, _, labels = process_batch(batch)
        for j in range(input_ids.shape[0]):
            id_ = i + j
            input_toks, label_toks = input_ids[j], labels[j].unsqueeze(0).int()
            output += [{
                'input_text': tokenizer.batch_decode([input_toks], skip_special_tokens=True)[0],
                'label_text': tokenizer.batch_decode([label_toks], skip_special_tokens=True)[0],
            }]
    with open(os.path.join(out_path, 'trak_test.json'), "w") as f:
        json.dump(output, f)


def main(ckpt, out, device='cuda'):
    out_path = os.path.join(out, RUN_NAME)

    model = CausalLM(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side='left',
        trust_remote_code=True
    )
    if isinstance(tokenizer, OLMoTokenizerFast):
        tokenizer.bos_token = tokenizer.eos_token

    loader_train, loader_val = init_loaders(tokenizer, batch_size=BATCH_SIZE)

    traker = TRAKer(
        model=model,
        task=CausalLMModelOutput,
        train_set_size=TRAIN_SET_SIZE,
        save_dir=out_path,
        device=device,
        proj_dim=1024
    )

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in tqdm(loader_train, desc='Featurizing'):
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

    for batch in tqdm(loader_val, desc='Scoring'):
        batch = process_batch(batch)
        batch = [x.to(device) for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name=RUN_NAME)

    print(scores)
    print(scores.shape)

    save_outputs(out_path, scores, loader_train, loader_val, tokenizer)


if __name__ == "__main__":
    """
    CUDA_VISIBLE_GPUS=0 nohup python trak_olmo.py > ../log/trak_olmo.log & tail -f ../log/trak_olmo.log 

    scp dheineman3@sky1.cc.gatech.edu:/nethome/dheineman3/nlprx/trak/results/lima-04-20-18-00-22/attribution_scores.pt /Users/dhei/personal/4644/dl/data/trak
    """
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='model checkpoint', default=None)
    parser.add_argument('--out', type=str, help='dir to save TRAK scores and metadata to', default='../results')
    args = parser.parse_args()
    main(args.ckpt, args.out)


def test_tokenizer(input_string, device='cuda'):
    model = CausalLM(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side='left',
        trust_remote_code=True
    )
    test_tok = tokenizer(
        input_string, 
        return_tensors='pt', 
        return_token_type_ids=False,
        padding=True
    )
    print(model(test_tok['input_ids'].to(device), test_tok['attention_mask'].to(device)))
