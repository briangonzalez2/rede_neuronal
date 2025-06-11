from utils import tokenize_code, extract_ast_features

# Código de ejemplo que queremos analizar
code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
"""

# Procesar el código con las funciones de utils.py
tokens = tokenize_code(code)
ast_nodes = extract_ast_features(code)

# Mostrar resultados
print("📦 Tokens (input_ids):")
print(tokens["input_ids"])

print("\n🌳 AST nodes:")
print(ast_nodes)
