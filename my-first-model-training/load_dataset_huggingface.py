from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Split into train and test
train_data = dataset["train"]
test_data = dataset["test"]

# Print the first example
print(train_data[0])
