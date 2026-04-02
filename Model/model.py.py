import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from transformers import DebertaModel, Wav2Vec2Model
from config import MODALITY_DROPOUT_P

class FeatureExtractors(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.audio_enc = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.visual_enc = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.visual_enc.fc = nn.Identity()

    def forward(self, text, audio, image):
        t = self.text_enc(**text).last_hidden_state[:, 0]
        a = self.audio_enc(**audio).last_hidden_state.mean(dim=1)
        v = self.visual_enc(image)
        return t, a, v


class MISLS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.t = nn.Linear(768, d)
        self.a = nn.Linear(768, d)
        self.v = nn.Linear(2048, d)

    def forward(self, t, a, v):
        return self.t(t), self.a(a), self.v(v)


class MaskedAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3)
        self.scale = d ** 0.5

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        w = torch.softmax((q @ k.transpose(-1, -2)) / self.scale, dim=-1)
        return w @ v, w


class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, bidirectional=True, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1]


class MultiTaskHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emo = nn.Linear(d * 2, 7)
        self.sent = nn.Linear(d * 2, 3)

    def forward(self, h):
        return self.emo(h), self.sent(h)


class FullModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ext = FeatureExtractors()
        self.misls = MISLS(d)
        self.attn = MaskedAttention(d)
        self.gru = BiGRU(d)
        self.head = MultiTaskHead(d)

    def forward(self, text, audio, img, mask):
        t, a, v = self.ext(text, audio, img)
        zt, za, zv = self.misls(t, a, v)

        if self.training:
            drop = (torch.rand(mask.size(0), 1, device=mask.device) > MODALITY_DROPOUT_P).float()
            mask = mask * torch.cat([torch.ones_like(drop), drop, drop], dim=1)

        zt, za, zv = zt * mask[:, 0:1], za * mask[:, 1:2], zv * mask[:, 2:3]
        fused = torch.stack([zt, za, zv], dim=1)
        out, _ = self.attn(fused)
        h = self.gru(out)
        return self.head(h)