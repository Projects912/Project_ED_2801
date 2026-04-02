import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from dataset import MELDDataset, emotion_encoder
from model import FullModel
from train import train_epoch, evaluate

if __name__ == "__main__":

    csv_dir = os.path.join(BASE_DIR, "csv")
    train_csv = os.path.join(csv_dir, "train_sent_emo.csv")
    val_csv = os.path.join(csv_dir, "val_sent_emo.csv")
    test_csv = os.path.join(csv_dir, "test_sent_emo.csv")

    train_ds = MELDDataset(train_csv, "train")
    val_ds   = MELDDataset(val_csv, "val")
    test_ds  = MELDDataset(test_csv, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = FullModel(LATENT_DIM).to(DEVICE)

    for p in model.ext.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    emo_counts = np.bincount(
        train_ds.df.Emotion.map(lambda x: emotion_encoder.transform([x])[0])
    )
    emo_weights = torch.tensor(1.0 / (emo_counts + 1e-6)).to(DEVICE)

    crit_emo = nn.CrossEntropyLoss(weight=emo_weights)
    crit_sent = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):

        if epoch == FREEZE_EPOCHS:
            for p in model.ext.parameters():
                p.requires_grad = True

        tr = train_epoch(model, train_loader, optimizer, crit_emo, crit_sent)
        va = evaluate(model, val_loader, crit_emo, crit_sent)

        print(f"Epoch {epoch+1} | Train F1: {tr[1]:.3f} | Val F1: {va[1]:.3f}")

        if va[1] > best_val_f1:
            best_val_f1 = va[1]
            torch.save(model.state_dict(), "best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))
    te = evaluate(model, test_loader, crit_emo, crit_sent)

    print(f"TEST F1: {te[1]:.3f} | UAR: {te[2]:.3f} | Sent F1: {te[3]:.3f}")