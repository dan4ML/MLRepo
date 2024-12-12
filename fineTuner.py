import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
from datetime import datetime
from tqdm import tqdm  # Import tqdm
import json

class TextToSQLTrainer:
    def __init__(self, model_path, train_data_path, val_data_path, tables_data_path, num_train_rows, num_val_rows):
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.tables_data_path = tables_data_path
        self.num_train_rows = num_train_rows
        self.num_val_rows = num_val_rows

        # Load the pretrained T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        # Load the Spider dataset
        self.train_data = self.load_spider_data(train_data_path, tables_data_path, num_rows=num_train_rows)
        self.val_data = self.load_spider_data(val_data_path, tables_data_path, num_rows=num_val_rows)

    def load_spider_data(self, file_path, tables_file_path, num_rows=None):
        """Load and preprocess Spider dataset from the given file path."""
        print("Loading data for: ", file_path)

        data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            spider_data = json.load(f)

        with open(tables_file_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)

        # Create a dictionary to map database names to their schema information
        db_schemas = {table['db_id']: table for table in tables_data}

        # Loop through each entry and format as {source, target}
        for i, entry in enumerate(spider_data):
            if num_rows != "all" and num_rows is not None and i >= int(num_rows):
                break

            question = entry['question']
            query = entry['query']
            db_id = entry['db_id']
            schema = db_schemas[db_id]

            # Format schema information
            schema_info = f"Database: {db_id}\n"
            for table in schema['table_names_original']:
                schema_info += f"Table: {table}\n"
                columns = [col[1] for col in schema['column_names_original'] if col[0] == schema['table_names_original'].index(table)]
                schema_info += f"Columns: {', '.join(columns)}\n"

            # Create a dictionary for each entry
            data.append({
                "source": f"question: {question} schema: {schema_info}",
                "target": query
            })

            print("question: ", question, " schema: ", schema_info, " query: ", query)

        return data

    class TextToSQLDataset(Dataset):
        def __init__(self, tokenizer, data, max_length=128):
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

    def calculate_accuracy(self, preds, labels):
        # Ensure labels are within the valid range
        valid_labels = labels.clone()
        valid_labels[valid_labels == -100] = self.tokenizer.pad_token_id

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(valid_labels, skip_special_tokens=True)

        correct = 0
        total = len(decoded_preds)

        for pred, label in zip(decoded_preds, decoded_labels):
            if pred.strip() == label.strip():
                correct += 1

        return correct / total

    def fine_tune_t5_model(self, learning_rate=5e-5, epochs=5, batch_size=2, accumulation_steps=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device (GPU/CPU)
        self.model = self.model.to(device)
        print("Cuda with GPU or CPU? =>", device)
        #print("model: ", self.model)

        # Enable DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Load the dataset
        train_dataset = self.TextToSQLDataset(self.tokenizer, self.train_data[:20])
        val_dataset = self.TextToSQLDataset(self.tokenizer, self.val_data[:20])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)

        # Optimizer and learning rate scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs // accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Mixed precision training
        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

        # Gradient clipping
        max_grad_norm = 1.0

        # Fine-tuning loop
        losses = []
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0
            total_accuracy = 0

            # Wrap the train_loader with tqdm
            for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss / accumulation_steps  # Scale the loss

                scaler.scale(loss).backward()

                # Gradient clipping
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
                total_loss += loss.item() * accumulation_steps  # Unscale the loss

                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=-1)
                accuracy = self.calculate_accuracy(preds, labels)
                total_accuracy += accuracy

            avg_train_loss = total_loss / len(train_loader)
            avg_train_accuracy = total_accuracy / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Training loss: {avg_train_loss}, Training accuracy: {avg_train_accuracy}")

            # Validation after each epoch
            self.model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                # Wrap the val_loader with tqdm
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask, labels=labels
                        )
                        val_loss += outputs.loss.item()

                    # Calculate accuracy
                    preds = torch.argmax(outputs.logits, dim=-1)
                    accuracy = self.calculate_accuracy(preds, labels)
                    val_accuracy += accuracy

            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation loss: {avg_val_loss}, Validation accuracy: {avg_val_accuracy}")
            print(f"Completed epoch {epoch + 1}/{epochs}")

        return self.model, losses

if __name__ == "__main__":
    dateStrStart = datetime.now().strftime('%Y%m%dT%H%M%S')
    print(f"{dateStrStart}: Starting ML Training...")

    # Define the model path
    # Windows
    model_path = 'windows/path/to/model/T5-3B'
    
    # Linux
    #model_path = 'linux/path/to/model/T5-3B'

    
    # Define the data paths
    #Windows
    train_data_path = 'windows/path/to/train_spider.json'
    val_data_path = 'windows/path/to//dev.json'
    tables_data_path = 'windows/path/to/tables.json'

    # Linux
    #train_data_path = 'linux/path/to/train_spider.json'
    #val_data_path = 'linux/path/to/dev.json'
    #tables_data_path = 'linux/path/to/tables.json'
    
    # Specify the number of rows to load. Use 'all' to load all the dataset.
    num_train_rows = '10'
    num_val_rows = '10'

    # Initialize the trainer
    trainer = TextToSQLTrainer(model_path, train_data_path, val_data_path, tables_data_path, num_train_rows, num_val_rows)

    # Fine-tune the model using Spider dataset
    fine_tuned_model, training_losses = trainer.fine_tune_t5_model()

    # Save the fine-tuned model
    # Windows
    output_dir = "windows/path/to/T5-3B-finetuned"
    
    # Linux
    #output_dir = "linux/path/to/T5-3B-finetuned"
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fine_tuned_model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")
    dateStrFinish = datetime.now().strftime('%Y%m%dT%H%M%S')
    print(f"{dateStrFinish}: ...Finished ML Training")
