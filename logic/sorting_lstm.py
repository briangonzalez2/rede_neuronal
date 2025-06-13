import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

class MultiSortDataset(Dataset):
    def __init__(self, path="dataset/multi_sort.json"):
        with open(path) as f:
            self.data = json.load(f)
        self.action_types = ["swap", "pivot", "split", "merge"]
        self.algo_types = ["bubble_sort", "insertion_sort", "selection_sort", "quick_sort", "merge_sort"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        state = torch.tensor(item["state"], dtype=torch.float32)
        algo_idx = self.algo_types.index(item["type"])
        action_type = [k for k in item["action"].keys()][0]
        action_idx = self.action_types.index(action_type)
        if action_type == "swap":
            action_value = item["action"]["swap"]
        elif action_type == "pivot" or action_type == "split":
            action_value = [item["action"][action_type], 0]
        else:  # merge
            action_value = [item["action"]["merge"][0], 0]  # Only position
        return state, torch.tensor([algo_idx, action_idx] + action_value, dtype=torch.long)

class SortingLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_algos=5, num_actions=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_algo = nn.Linear(hidden_size, num_algos)
        self.fc_action = nn.Linear(hidden_size, num_actions)
        self.fc_indices = nn.Linear(hidden_size, 2)  # For swap/pivot/split/merge positions
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dim
        _, (hn, _) = self.lstm(x)
        hn = hn.squeeze(0)
        return self.fc_algo(hn), self.fc_action(hn), self.fc_indices(hn)

def train_lstm():
    dataset = MultiSortDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SortingLSTM()
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(15):
        total_loss = 0
        for state, target in loader:
            optimizer.zero_grad()
            algo_out, action_out, indices_out = model(state)
            loss = (criterion(algo_out, target[:, 0]) +
                    criterion(action_out, target[:, 1]) +
                    mse(indices_out, target[:, 2:].float()))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "models/sorting_lstm.pt")
    return model

def predict_next_swap(model, state, algo_type):
    model.eval()
    dataset = MultiSortDataset()
    algo_idx = dataset.algo_types.index(algo_type)
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        algo_out, action_out, indices_out = model(state)
        pred_algo = algo_out.argmax(1).item()
        pred_action = action_out.argmax(1).item()
        indices = indices_out.round().long().tolist()[0]
    action_type = dataset.action_types[pred_action]
    if action_type in ["pivot", "split"]:
        action = {action_type: indices[0]}
    elif action_type == "merge":
        action = {"merge": [indices[0], 0]}  # Value placeholder
    else:
        action = {"swap": indices}
    return {"type": dataset.algo_types[pred_algo], "action": action}

if __name__ == "__main__":
    train_lstm()