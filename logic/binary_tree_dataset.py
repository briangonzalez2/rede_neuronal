import random
import json
import numpy as np

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def generate_random_bst(n_nodes=7):
    values = random.sample(range(1, 100), n_nodes)
    root = None
    for value in values:
        if root is None:
            root = TreeNode(value)
        else:
            current = root
            while True:
                if value < current.value:
                    if current.left is None:
                        current.left = TreeNode(value)
                        break
                    current = current.left
                else:
                    if current.right is None:
                        current.right = TreeNode(value)
                        break
                    current = current.right
    return root

def get_inorder(root, result):
    if root:
        get_inorder(root.left, result)
        result.append(root.value)
        get_inorder(root.right, result)
    return result

def tree_to_graph(root):
    nodes = []
    edges = []
    value_to_idx = {}
    idx = 0
    def traverse(node):
        nonlocal idx
        if node:
            value_to_idx[node.value] = idx
            nodes.append(node.value)
            idx += 1
            if node.left:
                edges.append([value_to_idx[node.value], idx])
                traverse(node.left)
            if node.right:
                edges.append([value_to_idx[node.value], idx])
                traverse(node.right)
    traverse(root)
    return nodes, edges

data = []
for _ in range(50000):
    root = generate_random_bst()
    nodes, edges = tree_to_graph(root)
    inorder = get_inorder(root, [])
    data.append({"nodes": nodes, "edges": edges, "inorder": inorder})

with open("dataset/binary_tree.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Generated {len(data)} trees in dataset/binary_tree.json")