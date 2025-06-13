import numpy as np
import json

def bubble_sort_steps(arr):
    arr = arr.copy()
    n = len(arr)
    steps = []
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                steps.append({"type": "bubble_sort", "state": arr.copy(), "action": {"swap": [j, j+1]}})
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return steps

def insertion_sort_steps(arr):
    arr = arr.copy()
    n = len(arr)
    steps = []
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            steps.append({"type": "insertion_sort", "state": arr.copy(), "action": {"swap": [j, j+1]}})
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return steps

def selection_sort_steps(arr):
    arr = arr.copy()
    n = len(arr)
    steps = []
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            steps.append({"type": "selection_sort", "state": arr.copy(), "action": {"swap": [i, min_idx]}})
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return steps

def quick_sort_steps(arr, low, high, steps):
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                steps.append({"type": "quick_sort", "state": arr.copy(), "action": {"swap": [i, j]}})
                arr[i], arr[j] = arr[j], arr[i]
        steps.append({"type": "quick_sort", "state": arr.copy(), "action": {"swap": [i+1, high]}})
        arr[i+1], arr[high] = arr[high], arr[i+1]
        return i + 1

    arr = arr.copy()
    if low < high:
        pi = partition(arr, low, high)
        steps.append({"type": "quick_sort", "state": arr.copy(), "action": {"pivot": pi}})
        quick_sort_steps(arr, low, pi-1, steps)
        quick_sort_steps(arr, pi+1, high, steps)
    return steps

def merge_sort_steps(arr):
    def merge(arr, left, mid, right, steps):
        left_arr = arr[left:mid+1]
        right_arr = arr[mid+1:right+1]
        i = j = 0
        k = left
        while i < len(left_arr) and j < len(right_arr):
            if left_arr[i] <= right_arr[j]:
                steps.append({"type": "merge_sort", "state": arr.copy(), "action": {"merge": [k, left_arr[i]]}})
                arr[k] = left_arr[i]
                i += 1
            else:
                steps.append({"type": "merge_sort", "state": arr.copy(), "action": {"merge": [k, right_arr[j]]}})
                arr[k] = right_arr[j]
                j += 1
            k += 1
        while i < len(left_arr):
            steps.append({"type": "merge_sort", "state": arr.copy(), "action": {"merge": [k, left_arr[i]]}})
            arr[k] = left_arr[i]
            i += 1
            k += 1
        while j < len(right_arr):
            steps.append({"type": "merge_sort", "state": arr.copy(), "action": {"merge": [k, right_arr[j]]}})
            arr[k] = right_arr[j]
            j += 1
            k += 1

    arr = arr.copy()
    steps = []
    def merge_sort(arr, left, right, steps):
        if left < right:
            mid = (left + right) // 2
            merge_sort(arr, left, mid, steps)
            merge_sort(arr, mid+1, right, steps)
            steps.append({"type": "merge_sort", "state": arr.copy(), "action": {"split": mid}})
            merge(arr, left, mid, right, steps)
    merge_sort(arr, 0, len(arr)-1, steps)
    return steps

data = []
for _ in range(200):  # 200 experiments per algorithm
    arr = np.random.randint(0, 100, 10).tolist()
    data.extend(bubble_sort_steps(arr))
    data.extend(insertion_sort_steps(arr))
    data.extend(selection_sort_steps(arr))
    data.extend(quick_sort_steps(arr, 0, len(arr)-1, []))
    data.extend(merge_sort_steps(arr))

with open("dataset/multi_sort.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Generated {len(data)} steps in dataset/multi_sort.json")