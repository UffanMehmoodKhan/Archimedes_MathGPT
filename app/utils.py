import torch

def collate_fn(batch, tokenizer, device):
    inputs = tokenizer([x['question'] for x in batch], padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer([x['answer'] for x in batch], padding=True, truncation=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100  # ignore padding
    return inputs.input_ids.to(device), inputs.attention_mask.to(device), labels.to(device)

def generate_answer(model, tokenizer, question, device, max_length=128):
    model.eval()
    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
