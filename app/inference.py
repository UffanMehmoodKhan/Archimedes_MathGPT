import torch
from app.model import load_model_and_tokenizer
from app.utils import generate_answer

def answer_question(question, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer("outputs/flan_t5_trained")
    model.to(device)

    answer = generate_answer(model, tokenizer, question, device)
    return answer
