import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

class StackQueueDataset(Dataset):
    def __init__(self, path="dataset/stack_queue.json"):
        with open(path) as f:
            self.data = json.load(f)

        self.op2idx = {"<pad>": 0, "<null>": 1, "push": 2, "pop": 3, "enqueue": 4, "dequeue": 5}
        self.val2idx = {"<pad>": 0, "<null>": 1}
        for i in range(1, 101):
            idx = i + 1  # starting at 2
            self.val2idx[str(i)] = idx

        self.vocab_size = max(max(self.op2idx.values()), max(self.val2idx.values())) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ops, vals, outs = [], [], []
        for step in item["sequence"]:
            op_name = step["op"].split("(")[0]
            if op_name not in self.op2idx:
                print(f"Invalid operation: {op_name}")
                ops.append(0)  # <pad>
                vals.append(1)  # <null>
                outs.append(1)  # <null>
                continue
            ops.append(self.op2idx.get(op_name, 0))

            val = step["value"]
            val_str = str(val) if val is not None else "<null>"
            if val is not None and not (1 <= val <= 100):
                print(f"Invalid value: {val}")
                val_str = "<null>"
            vals.append(self.val2idx.get(val_str, 1))

            out = step["output"]
            out_str = str(out) if out is not None else "<null>"
            outs.append(self.val2idx.get(out_str, 1))

        x_ops = torch.tensor(ops, dtype=torch.long)
        x_vals = torch.tensor(vals, dtype=torch.long)
        y = torch.tensor(outs, dtype=torch.long)
        if x_ops.max() >= self.vocab_size or x_vals.max() >= self.vocab_size:
            print(f"Invalid indices: ops={x_ops.tolist()}, vals={x_vals.tolist()}")
        return x_ops, x_vals, y

class StackQueueRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.embed_op = nn.Embedding(vocab_size, hidden_size)
        self.embed_val = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_ops, x_vals):
        e_ops = self.embed_op(x_ops)
        e_vals = self.embed_val(x_vals)
        x = torch.cat((e_ops, e_vals), dim=-1)
        out, _ = self.rnn(x)
        return self.fc(out)

def train_rnn():
    dataset = StackQueueDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = StackQueueRNN(vocab_size=dataset.vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for x_ops, x_vals, y in dataloader:
            optimizer.zero_grad()
            out = model(x_ops, x_vals)
            loss = criterion(out.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/stack_queue_rnn.pt")
    with open("models/vocab.json", "w") as f:
        json.dump({"op2idx": dataset.op2idx, "val2idx": dataset.val2idx}, f)
    return model

def predict_output(sequence, ds_type="stack"):
    with open("models/vocab.json") as f:
        vocab_data = json.load(f)
        op2idx = vocab_data["op2idx"]
        val2idx = vocab_data["val2idx"]

    vocab_size = max(max(op2idx.values()), max(val2idx.values())) + 1
    model = StackQueueRNN(vocab_size)
    model.load_state_dict(torch.load("models/stack_queue_rnn.pt"))
    model.eval()

    ops = []
    vals = []
    for op in sequence:
        name = op.split("(")[0]
        if name not in op2idx:
            raise ValueError(f"Invalid operation: {name}")
        if name in ["pop", "dequeue"]:
            if "(" in op and op != f"{name}()":
                raise ValueError(f"{name} should not have a value: {op}")
            vals.append(val2idx.get("<null>", 1))
        elif name in ["push", "enqueue"]:
            if not "(" in op or not ")" in op:
                raise ValueError(f"Invalid format for {name}: {op}")
            val = op.split("(")[1].replace(")", "")
            if not val.isdigit() or not (1 <= int(val) <= 100):
                raise ValueError(f"Invalid value for {name}: {val}")
            vals.append(val2idx.get(val, 1))
        ops.append(op2idx.get(name, 0))

    x_ops = torch.tensor([ops], dtype=torch.long)
    x_vals = torch.tensor([vals], dtype=torch.long)
    with torch.no_grad():
        out = model(x_ops, x_vals)
        preds = out.argmax(-1).squeeze(0).tolist()

    idx2val = {v: k for k, v in val2idx.items()}
    return [idx2val.get(p, "?") for p in preds]

if __name__ == "__main__":
    model = train_rnn()