import torch
from torch.utils.data import Dataset

import os
import numpy as np
import librosa

class VoiceDataset(Dataset):

    def __init__(self, male_dir: str, female_dir: str, nb_mfcc: int = 13) -> None:
        self.data = []
        self.labels = []
        self.nb_mfcc = nb_mfcc

        for file in os.listdir(male_dir):
            file_path = os.path.join(male_dir, file)
            mfcc = self.get_mfcc(file_path)
            self.data.append(mfcc)
            self.labels.append(0)
        print("Done for males")
        
        for file in os.listdir(female_dir):
            file_path = os.path.join(female_dir, file)
            mfcc = self.get_mfcc(file_path)
            self.data.append(mfcc)
            self.labels.append(1)
        print("Done for females")

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        return
    
    def get_mfcc(self, file_path: str) -> np.ndarray:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.nb_mfcc)
        return mfcc.mean(axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

if __name__ == "__main__":
    raise NotImplementedError