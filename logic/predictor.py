import torch
import sys
import os
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .tokenizer import ast_tokenizer
from .label_encoder import decode_O, decode_omega, decode_theta
from .algorithms_base import MultiOutputModel

def predict_complexity(code_snippet):
    print(f"Input code:\n{code_snippet}")
    try:
        ast.parse(code_snippet)
        if not any(kw in code_snippet for kw in ['def', 'for', 'while', 'if']):
            return {"error": "Please enter a valid algorithmic function (e.g., with def, for, while)"}
    except SyntaxError:
        return {"error": "Invalid Python code"}
    
    tokens = ast_tokenizer(code_snippet)
    print(f"Tokens (first 10): {tokens[:10]}, Sum: {sum(tokens)}")
    input_tensor = torch.tensor([tokens], dtype=torch.float32)
    model = MultiOutputModel(256, 256, 26)
    try:
        model.load_state_dict(torch.load("models/complexity_model.pt", map_location=torch.device("cpu")))
    except Exception as e:
        return {"error": f"Model loading failed: {e}"}
    model.eval()

    with torch.no_grad():
        out_O, out_omega, out_theta = model(input_tensor)
        print(f"Logits: O={out_O[0][:5]}, Ω={out_omega[0][:5]}, Θ={out_theta[0][:5]}")
        pred_O = decode_O(out_O.argmax().item())
        pred_omega = decode_omega(out_omega.argmax().item())
        pred_theta = decode_theta(out_theta.argmax().item())

    result = {
        "O": pred_O,
        "Ω": pred_omega,
        "Θ": pred_theta
    }
    print(f"Prediction: {result}")
    return result