from tqdm.auto import tqdm
import json, os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '../results/lima-04-20-18-00-22'
RESULT_PATH = '../results/bert-attribution'

class ScoresDataset(Dataset):
    def __init__(self, tokenizer, train_path, test_path, scores_path, limit=None):
        self.tokenizer = tokenizer

        with open(train_path, 'r') as train_file:
            train_data = json.load(train_file)

        with open(test_path, 'r') as test_file:
            test_data = json.load(test_file)

        scores_data = torch.load(scores_path)

        # Softmax the last dimension of the scores
        scores_norm = scores_data.permute(2, 1, 0).contiguous()
        scores_norm = scores_norm.view(scores_norm.size(0), -1)
        scores_norm = torch.nn.functional.softmax(scores_norm, dim=1)
        scores_norm = scores_norm.permute(1, 0)
        scores_data = scores_norm.view(scores_data.size())

        self.pairs_list = []

        self.input_text  = [entry['input_text'] for entry in train_data]
        self.labels_text = [entry['labels_text'] for entry in train_data]
        self.test_text   = [entry['input_text'] for entry in test_data]

        # Old dataloading scheme
        # self.scores     = [entry['scores'] for entry in train_data]
        # for i in range(len(self.test_text)): # testing data
        #     for j in range(len(self.input_text)): # training data
        #         toAdd = (self.input_text[j][0], self.labels_text[j][0], self.test_text[i][0], self.scores[j][i])
        #         self.pairs_list.append(toAdd)

        # New dataloading scheme
        print(f'Loaded data: \n  Train data of {len(train_data)} documents\n  Test data of {len(test_data)} examples\n  Scores file of shape {scores_data.shape}\n')
        count = 0
        for train_idx in tqdm(range(len(train_data)), desc="Expanding training data", unit='document', leave=True):
            labels = self.labels_text[train_idx]
            for token_idx, _ in enumerate(labels):
                for test_idx, _ in enumerate(test_data):
                    self.pairs_list += [(
                        self.input_text[train_idx],
                        self.labels_text[train_idx][token_idx],
                        self.test_text[test_idx],
                        scores_data[train_idx, token_idx, test_idx]
                    )]

                    # If a limit is specified, only return the first LIMIT examples
                    if limit and count > limit: return
                    count += 1
        print(f'Finished loading {count} examples!')

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        input1, input2, input3, score = self.pairs_list[idx]
        total_input = input1 + " " + input2 + " " + input3
        tokenized = self.tokenizer(
            total_input, 
            truncation=True, 
            padding='max_length', 
            max_length = 512, 
            return_tensors ='pt'
        )
        # ret_score = self.scores[idx]
        if len(tokenized['input_ids']) > 512:
            print("Uh oh! It's ✨ truncated ✨")
            print(tokenized)
        return tokenized, score


def main():
    out_path = os.path.join(RESULT_PATH)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = ScoresDataset(
        tokenizer,
        os.path.join(DATA_PATH, 'trak_train.json'), 
        os.path.join(DATA_PATH, 'trak_test.json'),
        os.path.join(DATA_PATH, 'attribution_scores.pt'),
        limit = 100 # 30_000 # 1_000_000  # The xl dataset has 85_196_800 examples
    )

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator(device='cpu'))

    train_data_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, generator=torch.Generator(device='cpu'))
    test_data_loader  = DataLoader(test_dataset,  batch_size=48, shuffle=True, generator=torch.Generator(device='cpu'))


    learning_rate = 5e-4
    num_epochs = 10

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(DEVICE)
    optimizer = AdamW(model.parameters(), learning_rate)

    # Training Loop
    model.train()
    total_losses = []

    for epoch in range(num_epochs):
        iter = tqdm(train_data_loader, desc=f"Epoch {epoch+1}", unit='batch', leave=True)

        for input, score in iter:
            inputs2 = {k: v.squeeze().to(DEVICE) for k,v in input.items()}
            score2 = score.float().to(DEVICE)

            output = model(**inputs2, labels=score2.unsqueeze(-1))
            
            loss = output.loss
            total_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter.set_description(f'Epoch {epoch+1}/{num_epochs}, Current Loss: {loss:.4f}, Average Loss: {(total_loss/len(train_data_loader)):.4f}')

        total_losses += [(total_loss/len(train_data_loader))]
        
        # Saving Model
        model.save_pretrained(out_path)
    
    print(f'All losses: {total_losses}')

    # Eval loop
    evaluate(model, test_data_loader)


def evaluate(model, dataloader):
    model.eval()

    overall_error = 0
    count = 0

    with torch.no_grad():
        iter_ = tqdm(dataloader, leave=True)
        for input, score in iter_:
            inputs2 = {k: v.squeeze().to(DEVICE) for k,v in input.items()}
            score2 = score.float().to(DEVICE)
            
            if (len(inputs2['input_ids'].shape) != 2):
                inputs2 = {k: v.reshape(1, inputs2['input_ids'].shape[0]) for k,v in inputs2.items()}

            output = model(**inputs2)
            cur_pred = output.logits.squeeze().to(DEVICE)
            target = score2.reshape(cur_pred.shape)

            cur_error = nn.functional.mse_loss(cur_pred, target, reduction='sum')
            overall_error += cur_error.item()
            count += target.shape[0]

    mse = overall_error / count
    print(f"Final MSE: {mse}")


if __name__ == '__main__':
    main()

    # model = BertForSequenceClassification.from_pretrained(out_path).to(DEVICE)
    # evaluate(model, test_data_loader)