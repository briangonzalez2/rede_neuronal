# save_dummy_model.py
import torch
import torch.nn as nn
import os

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 3)  # Simula salida para O, Ω, Θ

    def forward(self, x):
        return self.linear(x)

# Crear carpeta si no existe
os.makedirs("models", exist_ok=True)

# Instancia y guarda el modelo dummy
model = DummyModel()
torch.save(model.state_dict(), "models/complexity_model.pt")

print("Modelo dummy guardado en models/complexity_model.pt")
