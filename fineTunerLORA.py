import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
from extractJson import load_spider_data  # Import the function from the new module
from datetime import datetime
from tqdm import tqdm  # Import tqdm

class TextToSQLDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data[idx]['source']
        target_text = self.data[idx]['target']

        # Tokenizing source and target text
        source_encoding = self.tokenizer(
            source_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding tokens in loss calculation

        return {
            "input_ids": source_encoding["input_ids"].flatten(),
            "attention_mask": source_encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

def calculate_accuracy(preds, labels, tokenizer):
    # Ensure labels are within the valid range
    valid_labels = labels.clone()
    valid_labels[valid_labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)

    correct = 0
    total = len(decoded_preds)

    for pred, label in zip(decoded_preds, decoded_labels):
        if pred.strip() == label.strip():
            correct += 1

    return correct / total

class LoRALinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, rank=4):
        super(LoRALinear, self).__init__(in_features, out_features, bias)
        self.rank = rank
        self.A = torch.nn.Parameter(torch.randn(in_features, rank))
        self.B = torch.nn.Parameter(torch.randn(rank, out_features))
        self.scaling = 1 / (rank ** 0.5)

    def forward(self, x):
        lora_output = self.scaling * (x @ self.A @ self.B)
        return super(LoRALinear, self).forward(x) + lora_output

def replace_linear_with_lora(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LoRALinear(module.in_features, module.out_features, module.bias is not None, rank))
        else:
            replace_linear_with_lora(module, rank)

def fine_tune_t5_model(model, tokenizer, train_data, val_data, learning_rate=5e-5, epochs=15, batch_size=24, accumulation_steps=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply LoRA to the model
    replace_linear_with_lora(model)

    # Move model to device (GPU/CPU)
    model = model.to(device)
    print("Cuda with GPU or CPU? =>", device)
    print ("model: ",model)

    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load the dataset
    train_dataset = TextToSQLDataset(tokenizer, train_data)
    val_dataset = TextToSQLDataset(tokenizer, val_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, pin_memory=True, prefetch_factor=2)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # Gradient clipping
    max_grad_norm = 1.0

    # Fine-tuning loop
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        total_accuracy = 0

        # Wrap the train_loader with tqdm
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / accumulation_steps  # Scale the loss

            scaler.scale(loss).backward()

            # Gradient clipping
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps  # Unscale the loss

            # Calculate accuracy
            preds = torch.argmax(outputs.logits, dim=-1)
            accuracy = calculate_accuracy(preds, labels, tokenizer)
            total_accuracy += accuracy

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training loss: {avg_train_loss}, Training accuracy: {avg_train_accuracy}")

        # Validation after each epoch
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            # Wrap the val_loader with tqdm
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    val_loss += outputs.loss.item()

                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=-1)
                accuracy = calculate_accuracy(preds, labels, tokenizer)
                val_accuracy += accuracy

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation loss: {avg_val_loss}, Validation accuracy: {avg_val_accuracy}")
        print(f"Completed epoch {epoch + 1}/{epochs}")

    return model

if __name__ == "__main__":

    dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
    print(f"{dateStrStart}: Starting ML Training...")

    # Define the model path
    # Windows
    #model_path = 'C:/ML/Capstone/LLM_Interface/GoogleT5/T5-3B'

    # Linux
    model_path = '/MainML/eagle/ImageData/LLM/T5-3B'
    
    # Load the pretrained T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Load the Spider dataset
    # Windows
    #train_data_path = 'C:/ML/Capstone/LLM_Interface/Datasets/Spider/spider/evaluation_examples/examples/train_spiderSmall.json'
    #val_data_path = 'C:/ML/Capstone/LLM_Interface/Datasets/Spider/spider/evaluation_examples/examples/devSmall.json'  
    
    # Linux
    train_data_path = '/MainML/eagle/ImageData/LLM/Datasets/Spider/spider/evaluation_examples/examples/train_spider.json'
    val_data_path = '/MainML/eagle/ImageData/LLM/Datasets/Spider/spider/evaluation_examples/examples/dev.json'


    # Specify the number of rows to load. Use 'all' to load all the dataset.
    num_train_rows = 'all'
    num_val_rows =  'all'

    train_data = load_spider_data(train_data_path, num_rows=num_train_rows)
    val_data = load_spider_data(val_data_path, num_rows=num_val_rows)


    # Debugging: Print some samples from the loaded data
    print("Sample from train_data:", train_data[:2])
    print("Sample from val_data:", val_data[:2])

    # Fine-tune the model using Spider dataset
    fine_tuned_model = fine_tune_t5_model(model, tokenizer, train_data, val_data)

    # Save the fine-tuned model
    output_dir = "C:/ML/Capstone/LLM_Interface/GoogleT5/T5-3B-LORA"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fine_tuned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")
    dateStrFinish = datetime.now().strftime('%Y%m%dT%H%M%S')
    print(f"{dateStrFinish}: ...Finished ML Training")
