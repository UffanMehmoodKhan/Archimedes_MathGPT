from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

# -----------------------------
# 1. Load Dataset
# -----------------------------
def get_mathqa_dataset():
    dataset = load_dataset("miike-ai/mathqa")

    # MathQA only has a 'train' split, so create a small test split manually
    train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset  = train_test_split["test"]

    return train_dataset, test_dataset

# -----------------------------
# 2. Load Model + Tokenizer
# -----------------------------
def load_model_and_tokenizer(model_name="google/flan-t5-base"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model     = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# -----------------------------
# 3. Preprocessing
# -----------------------------
def preprocess_function(examples, tokenizer, max_input_length=128, max_target_length=64):
    # Prefix for T5
    inputs = ["solve: " + q for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Tokenize targets
    labels = tokenizer(examples["answer"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# -----------------------------
# 4. Main Training Function
# -----------------------------
def main():
    # Load dataset
    train_dataset, test_dataset = get_mathqa_dataset()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Tokenize datasets
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_test  = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Data collator (handles padding dynamically)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/flan_t5_mathqa",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_steps=50,
        fp16=False,  # set True if your GPU supports
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

