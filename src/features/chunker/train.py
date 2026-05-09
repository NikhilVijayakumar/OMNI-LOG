import torch
import torch.optim as optim
from tqdm import tqdm
import mlflow
import os
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from features.chunker.model import BiLSTM_CRF
from features.data.loader import get_dataloader
from features.data.constants import TAG_MAP, DEFAULT_MAX_SEQ_LEN

# Hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(data_dir, model_save_path, config=None):
    # 1. Setup Hyperparameters from Config or Defaults
    if config:
        embedding_dim = config.get('train', {}).get('embedding_dim', EMBEDDING_DIM)
        hidden_dim = config.get('train', {}).get('hidden_dim', HIDDEN_DIM)
        lr = config.get('train', {}).get('lr', LEARNING_RATE)
        epochs = config.get('train', {}).get('epochs', EPOCHS)
        batch_size = config.get('train', {}).get('batch_size', BATCH_SIZE)
    else:
        embedding_dim, hidden_dim, lr, epochs, batch_size = EMBEDDING_DIM, HIDDEN_DIM, LEARNING_RATE, EPOCHS, BATCH_SIZE

    # 2. Setup Data & Model
    full_loader, processor = get_dataloader(data_dir, batch_size=batch_size)
    vocab_size = len(processor.vocab)
    full_dataset = full_loader.dataset

    # 80/20 Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BiLSTM_CRF(
        vocab_size=vocab_size,
        tag_to_ix=TAG_MAP,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)


    # 2. MLflow Tracking
    mlflow.set_experiment("OMNI-LOG-Chunker")
    with mlflow.start_run():
        mlflow.log_params({
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size
        })

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in loop:
                # Prepare Inputs
                tokens = batch['tokens'].to(DEVICE)
                tags = batch['tags'].to(DEVICE)
                # Create mask: 1 for real tokens, 0 for <PAD>
                # TorchCRF requires the first timestep to always be unmasked
                mask = (tokens != 0)
                mask[:, 0] = True

                # Forward Pass
                model.zero_grad()
                loss = model(tokens, tags, mask=mask)

                # Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Prevent exploding gradients
                optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Validation Phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch['tokens'].to(DEVICE)
                    tags = batch['tags'].to(DEVICE)
                    mask = (tokens != 0)
                    mask[:, 0] = True
                    val_loss = model(tokens, tags, mask=mask)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(f"Epoch {epoch + 1} Avg Train Loss: {avg_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # 3. Save Artifacts
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': processor.vocab,
            'tag_map': TAG_MAP
        }, model_save_path)

        mlflow.log_artifact(model_save_path)
        print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train_model(data_dir="data/logs", model_save_path="models/chunker/best_model.pth")