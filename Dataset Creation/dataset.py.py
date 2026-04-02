import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DebertaTokenizer, Wav2Vec2Processor
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from config import BASE_DIR, FIXED_AUDIO_LEN

emotion_encoder = LabelEncoder()
emotion_encoder.fit([
    "anger", "disgust", "fear",
    "joy", "neutral", "sadness", "surprise"
])

sentiment_encoder = LabelEncoder()
sentiment_encoder.fit(["negative", "neutral", "positive"])


class MELDDataset(Dataset):
    def __init__(self, csv_path, split):
        self.df = pd.read_csv(csv_path)
        self.split = split

        self.tokenizer = DebertaTokenizer.from_pretrained(
            "microsoft/deberta-base"
        )
        self.audio_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"

        text = self.tokenizer(
            row.Utterance,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        # Audio
        audio_path = os.path.join(BASE_DIR, "audio", self.split, name + ".wav")
        audio_mask = 1
        try:
            wav, _ = sf.read(audio_path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        except:
            wav = np.zeros(FIXED_AUDIO_LEN)
            audio_mask = 0

        wav = torch.tensor(wav, dtype=torch.float32)
        wav = wav[:FIXED_AUDIO_LEN] if wav.size(0) > FIXED_AUDIO_LEN else \
              torch.nn.functional.pad(wav, (0, FIXED_AUDIO_LEN - wav.size(0)))

        audio = self.audio_processor(
            wav, sampling_rate=16000, return_tensors="pt"
        )

        # Image
        img_path = os.path.join(BASE_DIR, "frames", self.split, name + ".jpg")
        visual_mask = 1
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.img_tf(img)
        except:
            img = torch.zeros(3, 224, 224)
            visual_mask = 0

        modality_mask = torch.tensor(
            [1, audio_mask, visual_mask], dtype=torch.float32
        )

        emo = torch.tensor(
            emotion_encoder.transform([row.Emotion])[0],
            dtype=torch.long
        )
        sent = torch.tensor(
            sentiment_encoder.transform([row.Sentiment])[0],
            dtype=torch.long
        )

        return text, audio, img, modality_mask, emo, sent