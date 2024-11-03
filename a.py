from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


model_name = "cointegrated/rut5-base-multitask"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

from datasets import load_dataset
dataset = load_dataset('IlyaGusev/gazeta', revision="v2.0", trust_remote_code=True)

def preprocess_data(example):
    inputs = tokenizer(example["text"][:1000], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(example["summary"][:1000], padding="max_length", truncation=True, max_length=150)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()