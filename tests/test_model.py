import sys
import os
import pytest
import torch
import tempfile
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model

class Config:
    batch_size = 2
    seq_len = 5
    audio_feature_nb = 4
    d_model = 8
    n_heads = 2
    mask_values = [0.0,0.1]

@pytest.fixture(scope="module")
def config():
    return Config

@pytest.fixture(scope="module")
def batch_fixture(config):
    
    # Create a batch of random audio features
    audio_features = torch.randn(config.batch_size, config.seq_len, config.audio_feature_nb)
    #replace one sequence with zeros to simulate padding
    audio_features[0, config.seq_len-1, :] = 0
    padding_mask = (audio_features.abs().sum(dim=-1) != 0)  # True for non-padded positions

    # Create a skip intensity mask with 0 or 1 values
    skip_intensity_batch = torch.zeros(config.batch_size, config.seq_len)
    skip_intensity_batch[0,0] = 1
    return audio_features, skip_intensity_batch, padding_mask

@pytest.fixture(scope="module")
def config_fixture(config):
    config_data = {
        'model': {
            'd_model': config.d_model,
            'dropout': 0.1,
            'nheads': config.n_heads,
            'd_feed_forward': config.d_model*4,
            'num_layers': 6
        },
        'data': {
            'audio_feature_size': config.audio_feature_nb,
            'seq_max_len': config.seq_len
        },
        'mask': {
            'mask_values': config.mask_values
        }
    }
    temp_path = tempfile.mkdtemp()
    config_file = f"{temp_path}/config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return str(config_file)
    
def test_print_fixture(config,batch_fixture):
    print("config :")
    for attr, value in vars(config).items():
        if not attr.startswith("__"):
            print(f"{attr}: {value}")
    audio_features, skip_intensity_batch, padding_mask = batch_fixture
    print("\nAudio features : ", audio_features.shape,"\n" ,audio_features)
    print("\nSkip intensity batch : ", skip_intensity_batch.shape, "\n", skip_intensity_batch)
    print("\nPadding mask : ", padding_mask.shape, "\n", padding_mask)

def test_input_embeddings(config,batch_fixture):
    audio_features, skip_intensity_batch, padding_mask = batch_fixture

    embeddings_layer = model.InputEmbeddings(config.d_model, config.audio_feature_nb)
    embeddings = embeddings_layer(audio_features)

    print("\nInput embeddings example : ",embeddings.shape, "\n",embeddings)
    assert embeddings.shape == (config.batch_size, config.seq_len, config.d_model)
    

def test_positional_encoding(config,batch_fixture):
    audio_features, skip_intensity_batch, padding_mask = batch_fixture

    embeddings_layer = model.InputEmbeddings(config.d_model, config.audio_feature_nb)
    embeddings = embeddings_layer(audio_features)

    pos_encoding_layer = model.PositionalEncoding(config.d_model, dropout=0.1, seq_max_len=config.seq_len)
    pos_encoded_embeddings = pos_encoding_layer(embeddings)

    print("\nPositional encoding example : ",pos_encoded_embeddings.shape,"\n", pos_encoded_embeddings)
    assert pos_encoded_embeddings.shape == (config.batch_size, config.seq_len, config.d_model)
    

def test_attention_mask(config,config_fixture, batch_fixture):
    _,skip_intensity_batch, padding_mask = batch_fixture

    encoder = model.MusicRecoEncoder(config_fixture)
    attn_mask = encoder.attention_mask(skip_intensity_batch, padding_mask)

    print("\nAttention mask example : ",attn_mask.shape,"\n", attn_mask)
    assert attn_mask.shape == (config.n_heads * config.batch_size, config.seq_len, config.seq_len)
    

def test_encoder_forward(config,config_fixture, batch_fixture):
    audio_features, skip_intensity_batch, padding_mask = batch_fixture
    batch={"audio_features": audio_features, "skip_intensities": skip_intensity_batch, "padding_mask": padding_mask}

    encoder = model.MusicRecoEncoder(config_fixture)
    output = encoder(batch)

    print("\nEncoder output example : ",output.shape,"\n", output)
    assert output.shape == (config.batch_size, config.seq_len, config.audio_feature_nb)
    