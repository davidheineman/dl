import os, datetime, json
from argparse import ArgumentParser
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
from trak import TRAKer
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from hf_olmo import OLMoTokenizerFast

from utils.trak_output import CausalLMModelOutput
from utils.dataloader import load_tulu_dataset, load_hf_dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# MODEL_NAME = 'allenai/OLMo-1B'
MODEL_NAME = '/nethome/dheineman3/nlprx/trak/tulu/output/lima_1B'

RUN_NAME = 'lima' + datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")
# RUN_NAME = 'mmlu-04-17-22-07-46' # <- Original MMLU example
# RUN_NAME = 'lima-04-20-14-55-58' # <- Development example

TRAIN_FILE = '../data/tulu_v2_lima_only/tulu_v2_data.jsonl'

TRAIN_SET_SIZE  = 8192 # LIMA is ~1M tokens?
MAX_SEQ_LEN     = 256
VAL_SET_SIZE    = 4096 # MMLU test set is 14K examples
BATCH_SIZE      = 2


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
        self.model.train().cuda() # .eval()

    def forward(self, input_ids, attention_mask):
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        scores = output.logits
        # scores = scores[..., -1, 34:38] # Get the last token (-1), and only the 34:38 tokens
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

    # Save both datasets as a JSON, with all metadata
    output = []
    for i, batch in tqdm(enumerate(loader_train), desc='Saving scores over training data'):
        input_ids, _, labels = process_batch(batch)
        for j in range(input_ids.shape[0]):
            id_ = i + j
            input_toks, label_toks = input_ids[j], labels[j]
            output += [{
                'id': id_,
                'scores': scores[id_, :].tolist(),
                'input_text': tokenizer.batch_decode([input_toks], skip_special_tokens=True),
                'labels_text': tokenizer.batch_decode([label_toks], skip_special_tokens=True),
                'input_ids': input_toks.tolist(),
                'labels': label_toks.tolist()
            }]
    with open(os.path.join(out_path, 'results_trak_train.json'), "w") as f:
        json.dump(output, f, indent=4)

    output = []
    for i, batch in tqdm(enumerate(loader_val), desc='Saving test data'):
        input_ids, _, labels = process_batch(batch)
        for j in range(input_ids.shape[0]):
            id_ = i + j
            input_toks, label_toks = input_ids[j], labels[j]
            output += [{
                'id': id_,
                'input_text': tokenizer.batch_decode([input_toks], skip_special_tokens=True),
                # 'labels_text': tokenizer.batch_decode([torch.tensor(label_toks)], skip_special_tokens=True),
                'input_ids': input_toks.tolist(),
                'labels': label_toks.tolist()
            }]
    with open(os.path.join(out_path, 'results_trak_test.json'), "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
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
