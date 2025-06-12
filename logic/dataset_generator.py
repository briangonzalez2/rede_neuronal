import json
import random
import string
import libcst as cst
import sys
import os

from logic.tokenizer import ast_tokenizer
from logic.algorithms_base import algorithms

def generate_variation(code, name):
    try:
        tree = cst.parse_module(code)
        var_name = ''.join(random.choices(string.ascii_lowercase, k=5))
        wrapper = cst.MetadataWrapper(tree)
        transformer = cst.CSTTransformer()
        modified_tree = wrapper.visit(transformer)
        modified_code = modified_tree.code
        if random.choice([True, False]):
            if "for " in modified_code:
                modified_code = modified_code.replace("for ", "while ")
                modified_code = modified_code.replace("range(", "range(0, ")
                lines = modified_code.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("while "):
                        var = line.split("in")[0].split()[-1]
                        init_line = f"    {var} = 0"
                        inc_line = f"        {var} += 1"
                        lines[i] = line.replace(f"while {var} in", f"while {var} <")
                        lines.insert(i, init_line)
                        lines.insert(i + 2, inc_line)
                modified_code = "\n".join(lines)
        modified_code = modified_code.replace("arr", var_name)
        cst.parse_module(modified_code)
        return modified_code
    except:
        return code

def generate_dataset(num_variations=100):
    data = []
    for algo in algorithms:
        data.append({
            "code": algo["code"],
            "O": algo["O"],
            "Ω": algo["Ω"],
            "Θ": algo["Θ"]
        })
        for _ in range(num_variations):
            modified_code = generate_variation(algo["code"], algo["name"])
            if ast_tokenizer(modified_code) != [0] * 256:
                data.append({
                    "code": modified_code,
                    "O": algo["O"],
                    "Ω": algo["Ω"],
                    "Θ": algo["Θ"]
                })
    return data

if __name__ == "__main__":
    dataset = generate_dataset()
    with open("dataset/ast_dataset_ast.json", "w") as f:
        json.dump(dataset, f, indent=2)