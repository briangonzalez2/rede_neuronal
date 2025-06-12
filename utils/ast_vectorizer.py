import ast
from collections import Counter

class ASTVectorizer(ast.NodeVisitor):
    def __init__(self):
        self.counter = Counter()

    def generic_visit(self, node):
        self.counter[type(node).__name__] += 1
        super().generic_visit(node)

def extract_ast_features(code: str) -> dict:
    try:
        tree = ast.parse(code)
        visitor = ASTVectorizer()
        visitor.visit(tree)
        return dict(visitor.counter)
    except Exception as e:
        print("Error procesando AST:", e)
        return {}
