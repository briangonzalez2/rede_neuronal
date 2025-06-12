import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from training.model import ComplexityMLP

class ASTDataset(Dataset):
    def __init__(self, data, vec, encoder):
        self.data = data
        self.vec = vec
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.vec.transform([self.data[idx]["features"]]).toarray()[0]
        y = self.encoder.transform([self.data[idx]["O"]])[0]  # Solo O por ahora
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)

# Cargar dataset
with open("dataset/ast_dataset.json") as f:
    raw_data = json.load(f)

# Vectorizador y codificador
vectorizer = DictVectorizer()
features = vectorizer.fit_transform([d["features"] for d in raw_data])

label_encoder = LabelEncoder()
label_encoder.fit([d["O"] for d in raw_data])

# Dataset y modelo
dataset = ASTDataset(raw_data, vectorizer, label_encoder)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = ComplexityMLP(input_size=features.shape[1], num_classes=len(label_encoder.classes_))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Entrenamiento simple
for epoch in range(10):
    total = 0
    correct = 0
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += y.size(0)
        correct += (pred.argmax(dim=1) == y).sum().item()
    print(f"Epoch {epoch+1}: accuracy = {correct / total:.2f}")

# Guardar modelo y utilidades
torch.save(model.state_dict(), "models/complexity_model.pt")
import pickle
with open("models/vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
with open("models/label_encoder.pkl", "wb") as f: pickle.dump(label_encoder, f)
