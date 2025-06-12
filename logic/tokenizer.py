import ast
def ast_tokenizer(code):
    try:
        tree = ast.parse(code)
        tokens = [0] * 256
        depth = 0
        for node in ast.walk(tree):
            node_type = str(type(node).__name__)
            tokens[hash(node_type) % 128] += 1
            if hasattr(node, 'name'):
                tokens[hash(node.name) % 128 + 128] += 1
            if hasattr(node, 'op'):
                tokens[hash(str(type(node.op).__name__)) % 128 + 128] += 1
            depth = max(depth, sum(1 for _ in ast.walk(node)))
        tokens[255] = depth
        return [min(x, 10) for x in tokens]
    except:
        return [0] * 256