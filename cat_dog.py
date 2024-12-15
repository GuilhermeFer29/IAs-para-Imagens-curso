from datasets import load_dataset

database = load_dataset('cats_vs_dogs', split='train', streaming=True)

