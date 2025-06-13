import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, DataLoader
import json
import numpy as np

class BinaryTreeDataset:
    def __init__(self, path="dataset/binary_tree.json"):
        with open(path) as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor(item["nodes"], dtype=torch.float).unsqueeze(-1)
        edge_index = torch.tensor(item["edges"], dtype=torch.long).t().contiguous()
        y = torch.tensor(item["inorder"], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

class GNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(1, hidden_size)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 100)  # Predict values 1-100
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc(x)

def train_gnn():
    dataset = BinaryTreeDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = GNN()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out.view(-1, 100), data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "models/binary_tree_gnn.pt")
    return model

def predict_inorder(model, nodes, edges):
    model.eval()
    x = torch.tensor(nodes, dtype=torch.float).unsqueeze(-1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    with torch.no_grad():
        out = model(data)
        preds = out.argmax(-1).tolist()
    return preds

if __name__ == "__main__":
    train_gnn()