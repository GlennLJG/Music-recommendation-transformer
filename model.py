import torch
import torch.nn as nn
import math
import yaml

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, audio_feature_nb:int):
        super().__init__()
        self.d_model=d_model
        self.audio_features_embedding=nn.Linear(audio_feature_nb,d_model) #(B,seq_len, d_model)

    def forward(self, audio_features):
        embeddings=self.audio_features_embedding(audio_features)
        # Scale embeddings by sqrt(d_model)
        return embeddings*(self.d_model ** 0.5) #(B,seq_len,d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout:float, seq_max_len:int):
        super().__init__()
        self.d_model=d_model
        self.seq_max_len=seq_max_len
        self.dropout=nn.Dropout(dropout)

        # create a matrix of shape (seq_max_len, d_model)
        pe = torch.zeros(seq_max_len, d_model)
        # create a vector of shape (seq_max_len, 1)
        position=torch.arange(0,seq_max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*(-math.log(torch.tensor(10000.0))/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0) #(1, seq_max_len, d_model)
        # register pe as a buffer to avoid being considered as a model parameter
        self.register_buffer('pe',pe)

    def forward(self,x):
        # add positional encoding to the input embeddings
        # x shape: (B, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class MusicRecoEncoder(nn.Module):
    def __init__(self,config_path:str="config.yaml"):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # create embeddings layers
        self.src_emb=InputEmbeddings(self.config['model']['d_model'],self.config['data']['audio_feature_size'])

        # create positional encoding layers
        self.src_pos=PositionalEncoding(self.config['model']['d_model'], self.config['model']['dropout'], self.config['data']['seq_max_len'])

        # create encoder layers
        encoder_layer=nn.TransformerEncoderLayer(self.config['model']['d_model'], self.config['model']['nheads'], self.config['model']['d_feed_forward'], self.config['model']['dropout'],batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer, self.config['model']['num_layers'])

        # create final layer norm
        self.layer_norm=nn.LayerNorm(self.config['model']['d_model'])

        # create heads for prediction
        self.audio_feature_predictor_head=nn.Linear(self.config['model']['d_model'],self.config['data']['audio_feature_size'])

        self.register_buffer('mask_values',torch.tensor(self.config['mask']['mask_values'], dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        # initialize weights
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def attention_mask(self, skip_intensity_batch:torch.Tensor,padding_mask_batch:torch.Tensor):
        # Initialiser skip_vector avec des zéros
        skip_vector = torch.zeros_like(skip_intensity_batch, dtype=torch.float32, device=skip_intensity_batch.device)
        # Appliquer mask_values seulement aux positions valides (non-paddées)
        mask_values = torch.tensor(self.config['mask']['mask_values'], device=skip_intensity_batch.device)
        # Récupérer les indices valides
        valid_indices = skip_intensity_batch[padding_mask_batch].long()
        # Mapper mask_values sur les positions valides
        skip_vector[padding_mask_batch] = mask_values[valid_indices]
        # mean of the i,j values of the skip_vector
        # skip_vector: (batch_size, seq_len)
        # Résultat: (batch_size, seq_len, seq_len)
        attn_mask = -1 * (skip_vector.unsqueeze(2) * skip_vector.unsqueeze(1)) / 2
        attn_mask = attn_mask.masked_fill(~padding_mask_batch.unsqueeze(1), float('-inf'))
        # Répliquer pour chaque head d'attention
        # PyTorch attend (batch_size * num_heads, seq_len, seq_len)
        attn_mask = attn_mask.repeat(self.config['model']['nheads'], 1, 1)
        # Ajouter un masque causal pour le décodeur
        seq_len = skip_intensity_batch.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn_mask = attn_mask.masked_fill(causal_mask, float('-inf'))
        return attn_mask #(Batch, seq_len, seq_len)

    def forward(self, batch:dict):
        x = self.src_emb(batch['audio_features'])
        x = self.src_pos(x)

        attn_mask = self.attention_mask(batch['skip_intensities'], batch['padding_mask'])
        encoder_output = self.encoder(x, mask=attn_mask)
        encoder_output_norm = self.layer_norm(encoder_output)

        audio_feature_logits = self.audio_feature_predictor_head(encoder_output_norm)

        return audio_feature_logits