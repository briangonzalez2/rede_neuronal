import libcst as cst

class ASTTokenizer:
    def __init__(self, max_len=256):
        self.max_len = max_len
        self.vocab = {
            "Module": 1, "FunctionDef": 2, "For": 3, "While": 4, "If": 5,
            "Compare": 6, "BinOp": 7, "Add": 8, "Sub": 9, "Mult": 10,
            "Div": 11, "Assign": 12, "Name": 13, "Constant": 14, "Return": 15,
            "Call": 16, "List": 17, "Tuple": 18, "AugAssign": 19, "Break": 20,
            "Continue": 21, "Pass": 22, "Eq": 23, "NotEq": 24, "Lt": 25,
            "LtE": 26, "Gt": 27, "GtE": 28, "And": 29, "Or": 30, "Not": 31,
            "unknown": 0
        }

    def tokenize(self, code):
        try:
            tree = cst.parse_module(code)
            tokens = self._traverse(tree)
            tokens = tokens[:self.max_len]
            tokens += [0] * (self.max_len - len(tokens))
            return tokens
        except:
            return [0] * self.max_len

    def _traverse(self, node):
        tokens = []
        if isinstance(node, cst.CSTNode):
            node_type = node.__class__.__name__
            tokens.append(self.vocab.get(node_type, 0))
            if isinstance(node, cst.BinOp):
                op_type = node.operator.__class__.__name__
                tokens.append(self.vocab.get(op_type, 0))
            elif isinstance(node, cst.Compare):
                for op in node.comparators:
                    op_type = op.operator.__class__.__name__
                    tokens.append(self.vocab.get(op_type, 0))
            for child in node.children:
                tokens.extend(self._traverse(child))
        return tokens

def ast_tokenizer(code):
    tokenizer = ASTTokenizer()
    return tokenizer.tokenize(code)