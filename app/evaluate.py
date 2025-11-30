import torch
from torch.utils.data import DataLoader
from app.data import get_mathqa_dataset
from app.model import load_model_and_tokenizer
from app.utils import collate_fn, generate_answer

def evaluate_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = get_mathqa_dataset()
    tokenizer, model = load_model_and_tokenizer("outputs/flan_t5_trained")
    model.to(device)

    test_loader = DataLoader(test_data, batch_size=4,
                             collate_fn=lambda b: collate_fn(b, tokenizer, device))

    model.eval()
    total_loss = 0
    total_samples = 0
    for input_ids, attention_mask, labels in test_loader:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    print(f"Average test loss: {total_loss / total_samples:.4f}")

    # Sample inference
    sample_question = test_data[0]['question']
    print(f"\nSample question: {sample_question}")
    print(f"Predicted answer: {generate_answer(model, tokenizer, sample_question, device)}")
    print(f"Actual answer: {test_data[0]['answer']}")
