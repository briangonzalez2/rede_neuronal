# logic/dataset_generator.py

import random
import textwrap
from logic.algorithms_base import algorithms
from logic.utils import tokenize_code, extract_ast_features, preprocess_code


def randomize_code(code):
    lines = code.strip().split('\n')
    random.shuffle(lines)
    return '\n'.join(lines)

def generate_dataset(num_samples_per_algorithm=3):
    dataset = []

    for algo in algorithms:
        base_code = algo["code"]
        for _ in range(num_samples_per_algorithm):
            randomized_code = randomize_code(base_code)
            dedented_code = textwrap.dedent(randomized_code)

            tokens = tokenize_code(dedented_code)
            ast_nodes = extract_ast_features(dedented_code)

            sample = {
                "name": algo["name"],
                "tokens": tokens["input_ids"],
                "ast": ast_nodes,
                "O": algo["O"],
                "Ω": algo["Ω"],
                "Θ": algo["Θ"]
            }

            dataset.append(sample)

    return dataset
