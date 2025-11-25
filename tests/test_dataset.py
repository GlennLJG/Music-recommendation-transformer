import sys
import os
import pytest
import pandas as pd
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dataset

class Config:
    pad_idx = -1
    session_nb = 3
    seq_len = 5
    audio_feature_nb = 4
    idx_test = 0
    batch_size = 2

@pytest.fixture(scope="module")
def config():
    ConfigClass = Config
    cols = ['session_id', 'session_position', 'session_length', 'track_id_clean', 'skip_intensity'] + [f'feature_{i}' for i in range(ConfigClass.audio_feature_nb)]
    rows = []
    for session_id in range(ConfigClass.session_nb):
        session_real_len = random.randint(2, ConfigClass.seq_len)
        for position in range(1, session_real_len + 1):
            track_id = random.randint(0, ConfigClass.session_nb * ConfigClass.seq_len)

            feat_seed = random.Random(track_id)
            audio_features = [feat_seed.random()* 4 - 2 for _ in range(ConfigClass.audio_feature_nb)] # features between -2 and 2

            skip_intensity = random.randint(0, 1)

            row = [session_id, position, session_real_len, track_id, skip_intensity] + audio_features
            rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    ConfigClass.df = df
    return ConfigClass

def test_print_fixture(config):
    print("config :")
    for attr, value in vars(config).items():
        if not attr.startswith("__"):
            print(f"{attr}: {value}")

def test_MusicRecoDataset_init(config):#test the dataset initialization and padding
    padded_dataset = dataset.MusicRecoDataset(config.df, config.seq_len, config.pad_idx, config.audio_feature_nb)
    print("\naudio features array example : ",padded_dataset.audio_features_array.shape, "\n",padded_dataset.audio_features_array)
    assert (padded_dataset.audio_features_array.shape == (config.session_nb, config.seq_len, config.audio_feature_nb))
    assert (padded_dataset.audio_features_array.dtype == torch.float32)
    assert not torch.isnan(padded_dataset.audio_features_array).any(), "Audio features contain NaN values"
    print("\ntrack ids array example : ",padded_dataset.track_ids_array.shape, "\n",padded_dataset.track_ids_array)
    assert (padded_dataset.track_ids_array.shape == (config.session_nb, config.seq_len))
    print("\nskip intensity array example : ",padded_dataset.skip_intensity_array.shape, "\n",padded_dataset.skip_intensity_array)
    assert (padded_dataset.skip_intensity_array.shape == (config.session_nb, config.seq_len))
    print("\npadding mask array example : ",padded_dataset.padding_mask_array.shape, "\n",padded_dataset.padding_mask_array)
    assert (padded_dataset.padding_mask_array.shape == (config.session_nb, config.seq_len))
    assert (padded_dataset.padding_mask_array.dtype == torch.bool)
    #check that padding mask correspond to pad_idx in track_ids_array, skip_intensity_array and audio_features_array
    track_id_pad_mask = (padded_dataset.track_ids_array != config.pad_idx)
    skip_intensity_pad_mask = (padded_dataset.skip_intensity_array != config.pad_idx)
    audio_features_pad_mask = ~(padded_dataset.audio_features_array == config.pad_idx).all(dim=-1)
    assert torch.equal(padded_dataset.padding_mask_array, track_id_pad_mask), "Padding mask does not match track_ids_array"
    assert torch.equal(padded_dataset.padding_mask_array, skip_intensity_pad_mask), "Padding mask does not match skip_intensity_array"
    assert torch.equal(padded_dataset.padding_mask_array, audio_features_pad_mask), "Padding mask does not match audio_features_array"

def test_MusicRecoDataset_fct(config):
    padded_dataset = dataset.MusicRecoDataset(config.df, config.seq_len, config.pad_idx, config.audio_feature_nb)
    print("\nDataset length : ", len(padded_dataset))
    assert len(padded_dataset) == config.session_nb, "Dataset length does not match number of sessions"
    sample = padded_dataset[config.idx_test]
    print("\nSample item from dataset : ")
    for key, value in sample.items():
        print(f"{key}: shape {value.shape}\n{value}\n")
    assert sample['audio_features'].shape == (config.seq_len, config.audio_feature_nb)
    assert sample['track_ids'].shape == (config.seq_len,)
    assert sample['skip_intensities'].shape == (config.seq_len,)
    assert sample['padding_mask'].shape == (config.seq_len,)

def test_MusicRecoDataModule(config):
    padded_dataset = dataset.MusicRecoDataset(config.df, config.seq_len, config.pad_idx, config.audio_feature_nb)
    data_module = dataset.MusicRecoDataModule(padded_dataset)
    dataloader = data_module.get_dataloader(batch_size=config.batch_size, shuffle=False, num_workers=0)
    batch=next(iter(dataloader))
    print(f"\nExample of first batch : ")
    for key, value in batch.items():
        print(f"{key}: shape {value.shape}\n{value}\n")
    assert batch['audio_features'].shape == (config.batch_size, config.seq_len, config.audio_feature_nb)
    assert batch['track_ids'].shape == (config.batch_size, config.seq_len)
    assert batch['skip_intensities'].shape == (config.batch_size, config.seq_len)
    assert batch['padding_mask'].shape == (config.batch_size, config.seq_len)
    #print audio features shape for each batch 
    total_size = 0
    for i, batch in enumerate(dataloader):
        print(f"Batch {i} : {batch['audio_features'].shape}")
        total_size += batch['audio_features'].shape[0]
    assert total_size == len(padded_dataset), "Total size of batches does not match dataset"
        


