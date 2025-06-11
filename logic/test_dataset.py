# logic/test_dataset.py

from logic.dataset_generator import generate_dataset


dataset = generate_dataset(num_samples_per_algorithm=2)

for i, sample in enumerate(dataset):
    print(f"\n🧪 Sample {i + 1}")
    print(f"🔹 Name: {sample['name']}")
    print(f"📦 Tokens shape: {sample['tokens'].shape}")
    print(f"🌳 AST Nodes: {sample['ast'][:5]}... (total: {len(sample['ast'])})")
    print(f"📈 Complejidades: O={sample['O']} | Ω={sample['Ω']} | Θ={sample['Θ']}")