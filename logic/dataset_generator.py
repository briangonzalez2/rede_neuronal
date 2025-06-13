import json
import random
import string
import libcst as cst
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.tokenizer import ast_tokenizer
from logic.algorithms_base import algorithms

def generate_variation(code, name):
    print(f"Processing variation for {name}")
    try:
        var_names = {
            'arr': ''.join(random.choices(string.ascii_lowercase, k=5)),
            'i': ''.join(random.choices(string.ascii_lowercase, k=3)),
            'j': ''.join(random.choices(string.ascii_lowercase, k=3)),
            'n': ''.join(random.choices(string.ascii_lowercase, k=3)),
            'k': ''.join(random.choices(string.ascii_lowercase, k=3)),
            'low': ''.join(random.choices(string.ascii_lowercase, k=4)),
            'high': ''.join(random.choices(string.ascii_lowercase, k=4)),
            'mid': ''.join(random.choices(string.ascii_lowercase, k=4)),
            'node': ''.join(random.choices(string.ascii_lowercase, k=4)),
            'graph': ''.join(random.choices(string.ascii_lowercase, k=5)),
            'start': ''.join(random.choices(string.ascii_lowercase, k=5)),
            'x': ''.join(random.choices(string.ascii_lowercase, k=3)),
            'target': ''.join(random.choices(string.ascii_lowercase, k=6))
        }
        modified_code = code
        for old, new in var_names.items():
            modified_code = modified_code.replace(f" {old} ", f" {new} ")
            modified_code = modified_code.replace(f"({old})", f"({new})")
            modified_code = modified_code.replace(f"{old}=", f"{new}=")
            modified_code = modified_code.replace(f"{old}:", f"{new}:")
        if random.random() < 0.3:
            tree = cst.parse_module(modified_code)
            wrapper = cst.MetadataWrapper(tree)
            modified_code = wrapper.visit(cst.CSTTransformer()).code
        cst.parse_module(modified_code)
        tokens = ast_tokenizer(modified_code)
        if sum(tokens) > 10:
            return modified_code
        print(f"Variation for {name} rejected by tokenizer")
        return code
    except Exception as e:
        print(f"Error in variation for {name}: {e}")
        return code

def generate_dataset(num_variations=1000):
    start_time = time.time()
    data = []
    for idx, algo in enumerate(algorithms):
        print(f"Generating for algorithm {idx+1}/{len(algorithms)}: {algo['name']}")
        data.append({
            "code": algo["code"],
            "O": algo["O"],
            "Ω": algo["Ω"],
            "Θ": algo["Θ"]
        })
        valid_variations = 0
        for i in range(num_variations):
            if i % 20 == 0:
                print(f"  Variation {i+1}/{num_variations}")
            modified_code = generate_variation(algo["code"], algo["name"])
            if modified_code != algo["code"]:
                data.append({
                    "code": modified_code,
                    "O": algo["O"],
                    "Ω": algo["Ω"],
                    "Θ": algo["Θ"]
                })
                valid_variations += 1
                print(f"  Added variation {valid_variations}")
        print(f"  Added {valid_variations} valid variations")
    print(f"Generated {len(data)} snippets in {time.time() - start_time:.2f} seconds")
    return data

if __name__ == "__main__":
    dataset = generate_dataset()
    with open("dataset/ast_dataset_ast.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Dataset saved to dataset/ast_dataset_ast.json")