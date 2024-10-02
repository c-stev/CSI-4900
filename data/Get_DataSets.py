from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("ucsbnlp/liar",trust_remote_code=True)

# Specify the directory where you want to save the dataset
save_path = "./liar"

# Save the dataset to the specified directory
dataset.save_to_disk(save_path)

print(f"Dataset downloaded and saved locally at {save_path}")
