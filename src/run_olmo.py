from transformers import AutoModelForCausalLM, AutoTokenizer

def setup(model_name="allenai/OLMo-1B"):
    global model, tokenizer

    print(f'Loading model {model_name}...')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left'
    )


def generate(input_text):
    input_toks = tokenizer(
        input_text, 
        return_tensors='pt', 
        return_token_type_ids=False,
        padding=True
    )

    generation = model.generate(
        **input_toks, 
        max_new_tokens=1, 
        do_sample=False,
        top_p=1,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True
    )
    output = generation.sequences
    
    # Get token scores
    scores = generation.scores[0]
    print(scores.shape)

    # Convert all input tokens to [PAD] so they aren't returned
    input_length = input_toks['input_ids'][0].size(0)
    output[:, :input_length] = model.config.pad_token_id

    # Cut off all input tokens so only the generation is decoded
    output = output[:, input_length:]

    output_text = tokenizer.batch_decode(
        output, 
        skip_special_tokens=True
    )

    return output_text


if __name__ == '__main__':
    setup()
    # output = generate(["What is the capital of france? The capital is ", "My name is "])
    output = generate(["What is the capital of france? The capital is "])
    print(output)