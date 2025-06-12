import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from .tokenizer import ast_tokenizer
from .label_encoder import encode_labels, decode_O, decode_omega, decode_theta

algorithms = [
    {
        "name": "bubble_sort",
        "code": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
""",
        "O": "O(n^2)",
        "Ω": "Ω(n)",
        "Θ": "Θ(n^2)"
    },
    {
        "name": "binary_search",
        "code": """def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
        "O": "O(log n)",
        "Ω": "Ω(1)",
        "Θ": "Θ(log n)"
    },
    {
        "name": "linear_search",
        "code": """def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
""",
        "O": "O(n)",
        "Ω": "Ω(1)",
        "Θ": "Θ(n)"
    },
    {
        "name": "merge_sort",
        "code": """def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
""",
        "O": "O(n log n)",
        "Ω": "Ω(n log n)",
        "Θ": "Θ(n log n)"
    },
    {
        "name": "quick_sort",
        "code": """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",
        "O": "O(n^2)",
        "Ω": "Ω(n log n)",
        "Θ": "Θ(n log n)"
    },
    {
        "name": "insertion_sort",
        "code": """def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
""",
        "O": "O(n^2)",
        "Ω": "Ω(n)",
        "Θ": "Θ(n^2)"
    },
    {
        "name": "selection_sort",
        "code": """def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
""",
        "O": "O(n^2)",
        "Ω": "Ω(n^2)",
        "Θ": "Θ(n^2)"
    },
    {
        "name": "heap_sort",
        "code": """import heapq
def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]
""",
        "O": "O(n log n)",
        "Ω": "Ω(n log n)",
        "Θ": "Θ(n log n)"
    },
    {
        "name": "dfs",
        "code": """def dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                stack.append(neighbor)
    return visited
""",
        "O": "O(V + E)",
        "Ω": "Ω(V)",
        "Θ": "Θ(V + E)"
    },
    {
        "name": "factorial",
        "code": """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
""",
        "O": "O(n)",
        "Ω": "Ω(n)",
        "Θ": "Θ(n)"
    },
    {
        "name": "matrix_multiply",
        "code": """def matrix_multiply(A, B):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result
""",
        "O": "O(n^3)",
        "Ω": "Ω(n^3)",
        "Θ": "Θ(n^3)"
    },
    {
        "name": "fibonacci",
        "code": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
        "O": "O(2^n)",
        "Ω": "Ω(2^n)",
        "Θ": "Θ(2^n)"
    },
    {
        "name": "combination",
        "code": """def combination(n, k):
    if k == 0 or k == n:
        return 1
    return combination(n - 1, k - 1) + combination(n - 1, k)
""",
        "O": "O(nCk)",
        "Ω": "Ω(nCk)",
        "Θ": "Θ(nCk)"
    },
    {
        "name": "knapsack",
        "code": """def knapsack(values, weights, W):
    n = len(values)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]
""",
        "O": "O(nW)",
        "Ω": "Ω(nW)",
        "Θ": "Θ(nW)"
    },
    {
        "name": "sum_array",
        "code": """def sum_array(arr, k):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    for i in range(k):
        total += i
    return total
""",
        "O": "O(n + k)",
        "Ω": "Ω(n + k)",
        "Θ": "Θ(n + k)"
    },
    {
        "name": "permutations",
        "code": """def permutations(lst):
    if len(lst) <= 1:
        return [lst]
    result = []
    for i in range(len(lst)):
        n = lst.pop(0)
        perms = permutations(lst)
        for perm in perms:
            result.append([n] + perm)
        lst.append(n)
    return result
""",
        "O": "O(n!)",
        "Ω": "Ω(n!)",
        "Θ": "Θ(n!)"
    },
    {
        "name": "floyd_warshall",
        "code": """def floyd_warshall(graph):
    V = len(graph)
    dist = [[float('inf')] * V for _ in range(V)]
    for i in range(V):
        for j in range(V):
            dist[i][j] = graph[i][j]
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
""",
        "O": "O(V^3)",
        "Ω": "Ω(V^3)",
        "Θ": "Θ(V^3)"
    },
    {
        "name": "prim_mst",
        "code": """import heapq
def prim_mst(graph):
    V = len(graph)
    visited = [False] * V
    pq = [(0, 0)]
    total_weight = 0
    while pq:
        weight, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += weight
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (w, v))
    return total_weight
""",
        "O": "O(E log V)",
        "Ω": "Ω(E log V)",
        "Θ": "Θ(E log V)"
    },
    {
        "name": "strassen_matrix",
        "code": """def strassen_matrix(A, B):
    n = len(A)
    if n <= 1:
        return [[A[0][0] * B[0][0]]]
    mid = n // 2
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
    B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
    B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
    B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
    B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]
    P1 = strassen_matrix([[A11[i][j] + A22[i][j] for j in range(mid)] for i in range(mid)], [[B11[i][j] + B22[i][j] for j in range(mid)] for i in range(mid)])
    P2 = strassen_matrix([[A21[i][j] + A22[i][j] for j in range(mid)] for i in range(mid)], B11)
    P3 = strassen_matrix(A11, [[B12[i][j] - B22[i][j] for j in range(mid)] for i in range(mid)])
    P4 = strassen_matrix(A22, [[B21[i][j] - B11[i][j] for j in range(mid)] for i in range(mid)])
    P5 = strassen_matrix([[A11[i][j] + A12[i][j] for j in range(mid)] for i in range(mid)], B22)
    P6 = strassen_matrix([[A21[i][j] - A11[i][j] for j in range(mid)] for i in range(mid)], [[B11[i][j] + B12[i][j] for j in range(mid)] for i in range(mid)])
    P7 = strassen_matrix([[A12[i][j] - A22[i][j] for j in range(mid)] for i in range(mid)], [[B21[i][j] + B22[i][j] for j in range(mid)] for i in range(mid)])
    C11 = [[P1[i][j] + P4[i][j] - P5[i][j] + P7[i][j] for j in range(mid)] for i in range(mid)]
    C12 = [[P3[i][j] + P5[i][j] for j in range(mid)] for i in range(mid)]
    C21 = [[P2[i][j] + P4[i][j] for j in range(mid)] for i in range(mid)]
    C22 = [[P1[i][j] - P2[i][j] + P3[i][j] + P6[i][j] for j in range(mid)] for i in range(mid)]
    C = [[0] * n for _ in range(n)]
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]
    return C
""",
        "O": "O(n^2.807)",
        "Ω": "Ω(n^2.807)",
        "Θ": "Θ(n^2.807)"
    },
    {
        "name": "lcs",
        "code": """def lcs(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
""",
        "O": "O(nm)",
        "Ω": "Ω(nm)",
        "Θ": "Θ(nm)"
    }
]

class MultiOutputModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_O = nn.Linear(hidden_size // 2, output_size)
        self.output_omega = nn.Linear(hidden_size // 2, output_size)
        self.output_theta = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        features = self.shared(x)
        return (
            self.output_O(features),
            self.output_omega(features),
            self.output_theta(features)
        )

def load_dataset(path="dataset/ast_dataset_ast.json"):
    with open(path, "r") as f:
        data = json.load(f)
    X, y_O, y_omega, y_theta = [], [], [], []
    for item in data:
        tokens = ast_tokenizer(item["code"])
        if sum(tokens) > 0:
            X.append(tokens)
            o, omega, theta = encode_labels(item)
            y_O.append(o)
            y_omega.append(omega)
            y_theta.append(theta)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_O),
        torch.tensor(y_omega),
        torch.tensor(y_theta)
    )

def train():
    X, y_O, y_omega, y_theta = load_dataset()
    X_train, X_val, y_O_train, y_O_val, y_omega_train, y_omega_val, y_theta_train, y_theta_val = train_test_split(
        X, y_O, y_omega, y_theta, test_size=0.2
    )

    model = MultiOutputModel(256, 256, 26)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        out_O, out_omega, out_theta = model(X_train)
        loss_O = criterion(out_O, y_O_train)
        loss_omega = criterion(out_omega, y_omega_train)
        loss_theta = criterion(out_theta, y_theta_train)
        loss = loss_O + loss_omega + loss_theta
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc_O = (out_O.argmax(1) == y_O_train).float().mean()
        acc_omega = (out_omega.argmax(1) == y_omega_train).float().mean()
        acc_theta = (out_theta.argmax(1) == y_theta_train).float().mean()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc O: {acc_O:.2f}, Ω: {acc_omega:.2f}, Θ: {acc_theta:.2f}")

    torch.save(model.state_dict(), "models/complexity_model.pt")
    print("✅ Modelo multitarea guardado en models/complexity_model.pt")

if __name__ == "__main__":
    train()