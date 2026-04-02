import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm
from config import DEVICE

def train_epoch(model, loader, optimizer, crit_emo, crit_sent):
    model.train()
    losses, ye, pe, ys, ps = [], [], [], [], []

    for text, audio, img, mask, emo, sent in tqdm(loader):
        text = {k: v.squeeze(1).to(DEVICE) for k, v in text.items()}
        audio = {k: v.squeeze(1).to(DEVICE) for k, v in audio.items()}
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        emo, sent = emo.to(DEVICE), sent.to(DEVICE)

        optimizer.zero_grad()
        emo_out, sent_out = model(text, audio, img, mask)
        loss = crit_emo(emo_out, emo) + crit_sent(sent_out, sent)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        ye.extend(emo.cpu().numpy())
        pe.extend(torch.argmax(emo_out, 1).cpu().numpy())
        ys.extend(sent.cpu().numpy())
        ps.extend(torch.argmax(sent_out, 1).cpu().numpy())

    return (
        np.mean(losses),
        f1_score(ye, pe, average="macro"),
        recall_score(ye, pe, average="macro"),
        f1_score(ys, ps, average="macro")
    )


@torch.no_grad()
def evaluate(model, loader, crit_emo, crit_sent):
    model.eval()
    losses, ye, pe, ys, ps = [], [], [], [], []

    for text, audio, img, mask, emo, sent in loader:
        text = {k: v.squeeze(1).to(DEVICE) for k, v in text.items()}
        audio = {k: v.squeeze(1).to(DEVICE) for k, v in audio.items()}
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        emo, sent = emo.to(DEVICE), sent.to(DEVICE)

        emo_out, sent_out = model(text, audio, img, mask)
        loss = crit_emo(emo_out, emo) + crit_sent(sent_out, sent)

        losses.append(loss.item())
        ye.extend(emo.cpu().numpy())
        pe.extend(torch.argmax(emo_out, 1).cpu().numpy())
        ys.extend(sent.cpu().numpy())
        ps.extend(torch.argmax(sent_out, 1).cpu().numpy())

    return (
        np.mean(losses),
        f1_score(ye, pe, average="macro"),
        recall_score(ye, pe, average="macro"),
        f1_score(ys, ps, average="macro")
    )