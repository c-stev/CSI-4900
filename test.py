import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Define the path to your local model directory
model_path = "/home/wbeau097/CSI4900"

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Load the model and move it to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM.from_pretrained(model_path).to(device)

# Prompt the user for input
question = "What is the capital of Canada?"

# Tokenize the input
input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)

print(f"Model device: {next(model.parameters()).device}")
print(f"Input tensor device: {input_ids.device}")

# Generate output
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)