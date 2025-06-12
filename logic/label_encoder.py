label_map_O = {
    "O(1)": 0, "O(log n)": 1, "O(log log n)": 2, "O(n)": 3, "O(n log n)": 4,
    "O(n^2)": 5, "O(n^3)": 6, "O(n^k)": 7, "O(n^n)": 8, "O(2^n)": 9,
    "O(k^n)": 10, "O(n!)": 11, "O(n + k)": 12, "O(nk)": 13, "O(nCk)": 14,
    "O(k)": 15, "O(V)": 16, "O(E)": 17, "O(V + E)": 18, "O(n + m)": 19,
    "O(nW)": 20, "O(n!)": 21, "O(V^3)": 22, "O(E log V)": 23, "O(n^2.807)": 24,
    "O(nm)": 25
}

label_map_omega = {
    "Ω(1)": 0, "Ω(log n)": 1, "Ω(log log n)": 2, "Ω(n)": 3, "Ω(n log n)": 4,
    "Ω(n^2)": 5, "Ω(n^3)": 6, "Ω(n^k)": 7, "Ω(n^n)": 8, "Ω(2^n)": 9,
    "Ω(k^n)": 10, "Ω(n!)": 11, "Ω(n + k)": 12, "Ω(nk)": 13, "Ω(nCk)": 14,
    "Ω(k)": 15, "Ω(V)": 16, "Ω(E)": 17, "Ω(V + E)": 18, "Ω(n + m)": 19,
    "Ω(nW)": 20, "Ω(n!)": 21, "Ω(V^3)": 22, "Ω(E log V)": 23, "Ω(n^2.807)": 24,
    "Ω(nm)": 25
}

label_map_theta = {
    "Θ(1)": 0, "Θ(log n)": 1, "Θ(log log n)": 2, "Θ(n)": 3, "Θ(n log n)": 4,
    "Θ(n^2)": 5, "Θ(n^3)": 6, "Θ(n^k)": 7, "Θ(n^n)": 8, "Θ(2^n)": 9,
    "Θ(k^n)": 10, "Θ(n!)": 11, "Θ(n + k)": 12, "Θ(nk)": 13, "Θ(nCk)": 14,
    "Θ(k)": 15, "Θ(V)": 16, "Θ(E)": 17, "Θ(V + E)": 18, "Θ(n + m)": 19,
    "Θ(nW)": 20, "Θ(n!)": 21, "Θ(V^3)": 22, "Θ(E log V)": 23, "Θ(n^2.807)": 24,
    "Θ(nm)": 25
}

def encode_labels(item):
    return (
        label_map_O[item["O"]],
        label_map_omega[item["Ω"]],
        label_map_theta[item["Θ"]]
    )

def decode_O(index):
    for label, idx in label_map_O.items():
        if idx == index:
            return label

def decode_omega(index):
    for label, idx in label_map_omega.items():
        if idx == index:
            return label

def decode_theta(index):
    for label, idx in label_map_theta.items():
        if idx == index:
            return label