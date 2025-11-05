from load_tokenizer_model import tokenizer, model
text = "This movie was absolutely wonderful!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=1)
print("Prediction:", "Positive" if pred == 1 else "Negative")



