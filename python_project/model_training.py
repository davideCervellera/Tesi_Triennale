import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import random
from torch.optim import AdamW


# Impostazioni di setup per il training del modello tinyBERT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 2e-5
SEED = 42
MODEL_NAME = "prajjwal1/bert-tiny"


# Imposto i seed per la riprodubibilitá del codice

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



# Imposto il dataset nel giusto formato per fornire i dati al modello BERT

class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len=128):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = str(self.urls[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            url,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


df = pd.read_csv("normalized_dataset_urls.csv")
df = df.dropna(subset=['url', 'label'])
label_map = {'sicuro': 0, 'phishing': 1}
df['label'] = df['label'].map(label_map)

# Carico il tokenizer di BERT, imposto l'addestramento del modello con l'80 % dei dati, il restante 20% lo utilizzo per effettuare i test

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_texts, val_texts, train_labels, val_labels = train_test_split(df["url"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED)

train_dataset = URLDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = URLDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Carico il modello con dropout più alto
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.config.hidden_dropout_prob = 0.3
model.config.attention_probs_dropout_prob = 0.3
model.to(DEVICE)

# Imposto ottimizzatore, scheduler e funzione di perdita
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

loss_fct = torch.nn.CrossEntropyLoss()

# Inizio il training con funzione di Early Stopping (il training viene fermato se il modello non migliora)
# Per ogni epoca effettuo la validazione del modello
# Alla fine stampo il report sulla valutazione finale

best_val_loss = float('inf')
patience = 2
trigger_times = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fct(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validazione
    model.eval()
    total_val_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fct(outputs.logits, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        model.save_pretrained("trained_model")
        tokenizer.save_pretrained("trained_model")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stop attivato.")
            break

# Valutazione finale
print("\nValutazione finale:")
print(classification_report(true_labels, predictions, digits=4))

