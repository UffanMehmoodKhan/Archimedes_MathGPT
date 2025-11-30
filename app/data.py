from datasets import load_dataset, DatasetDict

def get_mathqa_dataset():
    dataset = load_dataset("miike-ai/mathqa")

    if "test" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.2)
        dataset = DatasetDict({"train": split["train"], "test": split["test"]})

    return dataset["train"], dataset["test"]
