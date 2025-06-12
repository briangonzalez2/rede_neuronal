import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from .tokenizer import simple_tokenizer
from .label_encoder import encode_labels, decode_O, decode_omega, decode_theta

# Modelo multitarea
class MultiOutputModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.output_O = nn.Linear(hidden_size, output_size)
        self.output_omega = nn.Linear(hidden_size, output_size)
        self.output_theta = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        features = self.shared(x)
        return (
            self.output_O(features),
            self.output_omega(features),
            self.output_theta(features)
        )

# Cargar y procesar el dataset
def load_dataset(path="dataset/ast_dataset_ast.json"):
    with open(path, "r") as f:
        data = json.load(f)
    X, y_O, y_omega, y_theta = [], [], [], []
    for item in data:
        tokens = simple_tokenizer(item["ast"])
        vector = [ord(c) for c in tokens][:256]
        vector += [0] * (256 - len(vector))
        o, omega, theta = encode_labels(item)
        X.append(vector)
        y_O.append(o)
        y_omega.append(omega)
        y_theta.append(theta)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_O),
        torch.tensor(y_omega),
        torch.tensor(y_theta)
    )

def train():
    X, y_O, y_omega, y_theta = load_dataset()
    X_train, X_val, y_O_train, y_O_val, y_omega_train, y_omega_val, y_theta_train, y_theta_val = train_test_split(
        X, y_O, y_omega, y_theta, test_size=0.2
    )

    model = MultiOutputModel(256, 128, 19)  # 19 posibles etiquetas
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        out_O, out_omega, out_theta = model(X_train)
        loss_O = criterion(out_O, y_O_train)
        loss_omega = criterion(out_omega, y_omega_train)
        loss_theta = criterion(out_theta, y_theta_train)
        loss = loss_O + loss_omega + loss_theta
        loss.backward()
        optimizer.step()

        acc_O = (out_O.argmax(1) == y_O_train).float().mean()
        acc_omega = (out_omega.argmax(1) == y_omega_train).float().mean()
        acc_theta = (out_theta.argmax(1) == y_theta_train).float().mean()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc O: {acc_O:.2f}, Ω: {acc_omega:.2f}, Θ: {acc_theta:.2f}")

    torch.save(model.state_dict(), "models/complexity_model.pt")
    print("✅ Modelo multitarea guardado en models/complexity_model.pt")

if __name__ == "__main__":
    train()
