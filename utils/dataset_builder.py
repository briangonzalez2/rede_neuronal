import os
import json
from ast_vectorizer import extract_ast_features

# Cambia según tus etiquetas reales
etiquetas_complejidad = {
    "bubble_sort.py": {"O": "O(n^2)", "Ω": "Ω(n)", "Θ": "Θ(n^2)"},
    "binary_search.py": {"O": "O(log n)", "Ω": "Ω(1)", "Θ": "Θ(log n)"},
    "radix_sort.py":      {"O": "O(nk)", "Ω": "Ω(nk)", "Θ": "Θ(nk)"},
    "insertion_sort.py":  {"O": "O(n^2)", "Ω": "Ω(n)", "Θ": "Θ(n^2)"},
    "counting_sort.py":   {"O": "O(n + k)", "Ω": "Ω(n + k)", "Θ": "Θ(n + k)"},
    "selection_sort.py": {"O": "O(n^2)", "Ω": "Ω(n^2)", "Θ": "Θ(n^2)"},
    "merge_sort.py": {"O": "O(n log n)", "Ω": "Ω(n log n)", "Θ": "Θ(n log n)"},
    "quick_sort.py": {"O": "O(n^2)", "Ω": "Ω(n log n)", "Θ": "Θ(n log n)"},
    "binary_search.py": {"O": "O(log n)", "Ω": "Ω(1)", "Θ": "Θ(log n)"},
    "linear_search.py": {"O": "O(n)", "Ω": "Ω(1)", "Θ": "Θ(n)"},
    "factorial_rec.py": {"O": "O(n)", "Ω": "Ω(n)", "Θ": "Θ(n)"},
    "fibonacci_rec.py": {"O": "O(2^n)", "Ω": "Ω(n)", "Θ": "Θ(2^n)"},
    "heap_sort.py": {"O": "O(n log n)", "Ω": "Ω(n log n)", "Θ": "Θ(n log n)"},
    "dfs_graph.py": {"O": "O(V + E)", "Ω": "Ω(V)", "Θ": "Θ(V + E)"},
    "matrix_multiplication.py": {"O": "O(n^3)", "Ω": "Ω(n^3)", "Θ": "Θ(n^3)"}
}

def crear_dataset(carpeta_codigo: str, output_path: str):
    dataset = []

    for filename in os.listdir(carpeta_codigo):
        if filename.endswith(".py"):
            path = os.path.join(carpeta_codigo, filename)
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            vector = extract_ast_features(code)
            labels = etiquetas_complejidad.get(filename, {})
            if labels:
                dataset.append({
                    "filename": filename,
                    "features": vector,
                    "O": labels["O"],
                    "Ω": labels["Ω"],
                    "Θ": labels["Θ"]
                })

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(dataset, out, indent=4)

if __name__ == "__main__":
    crear_dataset("data/codigo_ejemplos", "dataset/ast_dataset.json")
