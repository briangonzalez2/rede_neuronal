import libcst as cst
import ast
from transformers import AutoTokenizer

# Cargar el tokenizer de CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Función para convertir código en tokens (embeddings)
def tokenize_code(code):
    tokens = tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    return tokens

# AST Visitor
class ASTNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []

    def generic_visit(self, node):
        self.nodes.append(type(node).__name__)
        super().generic_visit(node)

# Función para extraer nodos AST
def extract_ast_features(code):
    try:
        tree = ast.parse(code)
        visitor = ASTNodeVisitor()
        visitor.visit(tree)
        return visitor.nodes
    except Exception as e:
        print(f"[❌ ERROR] AST parsing failed: {e}")
        return []
    
# Función para preprocesar el código
def preprocess_code(code):
    # Aquí podrías usar tree-sitter o ASTs para limpieza estructural
    return code.strip().replace("\n", " ")
