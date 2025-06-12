import torch
from .tokenizer import ast_tokenizer
from .label_encoder import decode_O, decode_omega, decode_theta
from .algorithms_base import MultiOutputModel

def predict_complexity(code_snippet):
    tokens = ast_tokenizer(code_snippet)
    input_tensor = torch.tensor([tokens], dtype=torch.float32)
    model = MultiOutputModel(256, 128, 19)
    model.load_state_dict(torch.load("models/complexity_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out_O, out_omega, out_theta = model(input_tensor)
        pred_O = decode_O(out_O.argmax().item())
        pred_omega = decode_omega(out_omega.argmax().item())
        pred_theta = decode_theta(out_theta.argmax().item())

    return {
        "O": pred_O,
        "Ω": pred_omega,
        "Θ": pred_theta
    }