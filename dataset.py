import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import math

class MusicRecoDataset(Dataset):
    def __init__(self, df, seq_len, pad_idx, nb_audio_features):
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.nb_of_sequences = df['session_id'].nunique()
        self.nb_audio_features = nb_audio_features

        # Initialisation numpy
        audio_features_array = np.full((self.nb_of_sequences, seq_len, nb_audio_features), pad_idx, dtype=np.float32)
        track_ids_array = np.full((self.nb_of_sequences, seq_len), pad_idx, dtype=np.int64)
        skip_intensity_array = np.full((self.nb_of_sequences, seq_len), pad_idx, dtype=np.int64)

        for i, (session_id, group) in enumerate(df.groupby('session_id')):
            group = group.sort_values('session_position')
            positions = (group['session_position'].to_numpy()) - 1  # zero-indexed
            track_ids_array[i, positions] = group['track_id_clean'].to_numpy()
            skip_intensity_array[i, positions] = group['skip_intensity'].to_numpy()
            audio_features_array[i, positions, :] = np.stack(group.iloc[:, -self.nb_audio_features:].to_numpy())

        # Conversion en tensor une fois
        self.audio_features_array = torch.tensor(audio_features_array, dtype=torch.float32)
        self.track_ids_array = torch.tensor(track_ids_array, dtype=torch.long)
        self.skip_intensity_array = torch.tensor(skip_intensity_array, dtype=torch.long)
        self.padding_mask_array = (self.track_ids_array != pad_idx)

    def __len__(self):
        return self.nb_of_sequences

    def __getitem__(self, idx):
        return {
            "audio_features": self.audio_features_array[idx],
            "track_ids": self.track_ids_array[idx],
            "skip_intensities": self.skip_intensity_array[idx],
            "padding_mask": self.padding_mask_array[idx]
        }

class MusicRecoDataModule:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_dataloader(self, batch_size, shuffle, num_workers):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)