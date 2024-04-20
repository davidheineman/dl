import os, datetime
from argparse import ArgumentParser
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
from trak import TRAKer
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from hf_olmo import OLMoTokenizerFast

from utils.trak_output import CausalLMModelOutput
from utils.dataloader import load_tulu_dataset, get_dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# MODEL_NAME = 'allenai/OLMo-1B'
MODEL_NAME = '/nethome/dheineman3/nlprx/trak/tulu/output/lima_1B'

RUN_NAME = 'lima' + datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")
# RUN_NAME = 'mmlu-04-17-22-07-46'

TRAIN_FILE = '../data/tulu_v2_lima_only/tulu_v2_data.jsonl'

TRAIN_SET_SIZE  = 128
VAL_SET_SIZE    = 16
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

    # /srv/nlprx-lab/share6/dheineman3/miniconda3/envs/trak/lib/python3.10/site-packages/hf_olmo/modeling_olmo.py
    def forward(self, input_ids, attention_mask):
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        scores = output.logits
        # scores = scores[..., -1, 34:38] # Get the last token (-1), and only the 34:38 tokens
        scores = scores[..., -1, :] # Get the last token (-1)
        scores = scores # Attribution on all tokens and logits!
        return scores


def init_loaders(tokenizer, batch_size):
    # Corresponds to the HF dataset split
    ds_train = load_tulu_dataset('train', tokenizer, TRAIN_FILE, limit=TRAIN_SET_SIZE, overwrite_cache=True) 
    # ds_train   = get_dataset('auxilary_train', tokenizer, limit=VAL_SET_SIZE) # auxilary_train, dev, test, validation
    ds_val   = get_dataset('validation', tokenizer, limit=VAL_SET_SIZE) # auxilary_train, dev, test, validation

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

    print(scores)
    print(scores.shape)


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
