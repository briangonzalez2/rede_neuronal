# Este script lo puedes ejecutar para guardar un modelo dummy
import torch
from logica.predictor import DummyModel

model = DummyModel()
torch.save(model.state_dict(), "models/complexity_model.pt")
