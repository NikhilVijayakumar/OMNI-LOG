import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow


class TripletLogDataset(Dataset):
    """
    Custom Dataset that produces (Anchor, Positive, Negative) triplets.
    """

    def __init__(self, logs, templates, processor):
        self.logs = logs  # List of raw logs
        self.templates = templates  # List of corresponding correct templates
        self.processor = processor
        self.unique_templates = list(set(templates))

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        anchor_text = self.logs[idx]
        pos_text = self.templates[idx]

        # Hard/Random Negative Mining: Pick a template that ISN'T the correct one
        neg_text = pos_text
        while neg_text == pos_text:
            neg_text = torch.utils.data.random_split(self.unique_templates, [1, len(self.unique_templates) - 1])[0][0]

        # Numericalize all three
        a_ids, _, a_len = self.processor.numericalize(self.processor.tokenize(anchor_text), [])
        p_ids, _, p_len = self.processor.numericalize(self.processor.tokenize(pos_text), [])
        n_ids, _, n_len = self.processor.numericalize(self.processor.tokenize(neg_text), [])

        return {
            "anchor": a_ids, "anchor_mask": (a_ids != 0),
            "pos": p_ids, "pos_mask": (p_ids != 0),
            "neg": n_ids, "neg_mask": (n_ids != 0)
        }


def train_siamese(encoder, train_loader, epochs=5, margin=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)  # p=2 for Euclidean distance

    mlflow.set_experiment("OMNI-LOG-Siamese")
    with mlflow.start_run():
        for epoch in range(epochs):
            encoder.train()
            epoch_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                # Encode all three parts of the triplet
                v_a = encoder(batch["anchor"].to(device), batch["anchor_mask"].to(device))
                v_p = encoder(batch["pos"].to(device), batch["pos_mask"].to(device))
                v_n = encoder(batch["neg"].to(device), batch["neg_mask"].to(device))

                loss = criterion(v_a, v_p, v_n)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("siamese_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch + 1} Siamese Loss: {avg_loss:.4f}")

    return encoder