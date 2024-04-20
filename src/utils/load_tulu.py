import torch
import numpy as np

def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    # Right padding to max sequence length
    features = {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
    features = pad(tokenizer, [features], padding='max_length', max_length=max_seq_length)

    # Fix tensor setup to 1D
    features = {
        'input_ids': features['input_ids'].flatten(),
        'labels': features['labels'].flatten(),
        'attention_mask': features['attention_mask'].flatten(),
    }
    
    features['labels'] = features['labels'][..., -2]
    features['input_ids'] = features['labels'][:max_seq_length]
    features['attention_mask'] = features['labels'][:max_seq_length]

    return features


def pad(tokenizer, features, 
        padding="longest", 
        label_pad_token_id=-100, 
        pad_to_multiple_of=None, 
        max_length=None,
        return_tensors="pt"
        ):
    if return_tensors is None:
        return_tensors = return_tensors
    
    labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
    
    if labels is None:
        raise ValueError("In my world, no empty labels!")
    
    # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    # same length to return tensors.
    if labels is not None:
        # max_label_length = max(len(l) for l in labels)
        max_label_length = max_length if max_length else max(len(l) for l in labels)

        # if pad_to_multiple_of is not None:
        #     max_label_length = (
        #         (max_label_length + pad_to_multiple_of - 1)
        #         // pad_to_multiple_of
        #         * pad_to_multiple_of
        #     )

        padding_side = tokenizer.padding_side
        for feature in features:
            remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
            else:
                feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

    features = tokenizer.pad(
        features,
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
    )

    # prepare decoder_input_ids
    # if (
    #     labels is not None
    #     and model is not None
    #     and hasattr(model, "prepare_decoder_input_ids_from_labels")
    # ):
    #     decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
    #     features["decoder_input_ids"] = decoder_input_ids

    return features
