from load_tokenizer_model import tokenizer
from load_dataset_huggingface import train_data, test_data

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)


