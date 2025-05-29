import torch
from transformers import AutoTokenizer, AutoModel
from logica.utils import preprocess_code

# Cargar modelo (simulado como una red densa)
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 3)  # O, Omega, Theta

    def forward(self, x):
        return self.linear(x)

# Modelo dummy (reemplaza con tu modelo real)
model = DummyModel()
model.load_state_dict(torch.load("models/complexity_model.pt", map_location="cpu"))
model.eval()

# Tokenizador simulado (reemplaza con tokenizer real)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert = AutoModel.from_pretrained("microsoft/codebert-base")

LABELS = ["O(n)", "Ω(n)", "Θ(n)"]  # ejemplo básico

def predict_complexity(code):
    cleaned = preprocess_code(code)
    tokens = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = codebert(**tokens).last_hidden_state.mean(dim=1)
        logits = model(embeddings)
        probs = torch.nn.functional.softmax(logits, dim=1)
    top_preds = torch.argmax(probs, dim=1)
    return {
        "O": LABELS[top_preds[0].item()],
        "Ω": LABELS[top_preds[0].item()],
        "Θ": LABELS[top_preds[0].item()]
    }
