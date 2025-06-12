import ast
import json
import os

INPUT = "dataset/ast_dataset.json"
OUTPUT = "dataset/ast_dataset_ast.json"

def extract_ast(source_code):
    try:
        tree = ast.parse(source_code)
        return ast.dump(tree)
    except Exception as e:
        print(f"Error al procesar:\n{source_code}\n{e}")
        return None

def build():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    for item in data:
        codigo = item["filename"]
        ast_text = extract_ast(codigo)
        if ast_text:
            processed.append({
                "codigo": codigo,
                "ast": ast_text,
                "O": item["O"],
                "Ω": item["Ω"],
                "Θ": item["Θ"]
            })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)
    print(f"✅ AST dataset generado con {len(processed)} ejemplos.")

if __name__ == "__main__":
    build()
