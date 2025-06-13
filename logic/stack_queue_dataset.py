import random
import json
import os

def generate_stack_sequence(n_ops=10):
    stack = []
    sequence = []
    for _ in range(n_ops):
        if random.random() < 0.6 or not stack:
            val = random.randint(1, 100)
            if not (1 <= val <= 100):
                print(f"Invalid stack value: {val}")
                continue
            stack.append(val)
            sequence.append({"op": f"push({val})", "output": None, "value": val})
        else:
            if stack:
                val = stack.pop()
                sequence.append({"op": "pop()", "output": val, "value": None})
    return {"type": "stack", "sequence": sequence}

def generate_queue_sequence(n_ops=10):
    queue = []
    sequence = []
    for _ in range(n_ops):
        if random.random() < 0.6 or not queue:
            val = random.randint(1, 100)
            if not (1 <= val <= 100):
                print(f"Invalid queue value: {val}")
                continue
            queue.append(val)
            sequence.append({"op": f"enqueue({val})", "output": None, "value": val})
        else:
            if queue:
                val = queue.pop(0)
                sequence.append({"op": "dequeue()", "output": val, "value": None})
    return {"type": "queue", "sequence": sequence}

os.makedirs("dataset", exist_ok=True)
data = []
for _ in range(1000):
    stack_seq = generate_stack_sequence()
    if stack_seq["sequence"]:
        data.append(stack_seq)
    queue_seq = generate_queue_sequence()
    if queue_seq["sequence"]:
        data.append(queue_seq)

with open("dataset/stack_queue.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Generated {len(data)} sequences in dataset/stack_queue.json")