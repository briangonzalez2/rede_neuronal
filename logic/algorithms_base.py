# logic/algorithms_base.py

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
    }
]
