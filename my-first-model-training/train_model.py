from load_tokenizer_model import model
from tokenize_dataset import tokenized_train, tokenized_test
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train.shuffle(seed=42).select(range(2000)),  # small subset for demo
    eval_dataset=tokenized_test.shuffle(seed=42).select(range(500)),
)

trainer.train()
