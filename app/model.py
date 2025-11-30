from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model_and_tokenizer(model_name="google/flan-t5-base"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model
