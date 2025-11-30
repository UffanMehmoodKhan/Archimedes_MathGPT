import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # <-- fixed import
from app.data import get_mathqa_dataset
from app.model import load_model_and_tokenizer
from app.utils import collate_fn

def train_model(epochs=3, batch_size=4, lr=5e-5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = get_mathqa_dataset()
    tokenizer, model = load_model_and_tokenizer()
    model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, device))

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, labels in train_loader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{epochs} completed. Last batch loss: {loss.item():.4f}")

    model.save_pretrained("outputs/flan_t5_trained")
    tokenizer.save_pretrained("outputs/flan_t5_trained")
    print("Training completed. Model saved to outputs/flan_t5_trained.")
