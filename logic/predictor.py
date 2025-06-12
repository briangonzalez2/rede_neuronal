import torch
from tokenizer import simple_tokenizer
from label_encoder import decode_O, decode_omega, decode_theta
from train_model import MultiOutputModel

def predict_complexity(code_snippet):
    tokens = simple_tokenizer(code_snippet)
    vector = [ord(c) for c in tokens][:256]
    vector += [0] * (256 - len(vector))

    input_tensor = torch.tensor([vector], dtype=torch.float32)
    model = MultiOutputModel(256, 128, 8)
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

