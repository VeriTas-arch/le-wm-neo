from einops import rearrange
from torch import nn


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, embed_dim=192):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = nn.Embedding(3, embed_dim)

        self.cue_probe = nn.Linear(embed_dim, 3)
        self.action_probe = nn.Linear(embed_dim, 3)

    def encode(self, info):
        pixels = info["pixels"].float()
        b, t = pixels.size(0), pixels.size(1)
        pixels = rearrange(pixels, "b t c h w -> (b t) c h w")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = rearrange(pixels_emb, "(b t) d -> b t d", b=b, t=t)
        info["emb"] = emb

        if "action" in info:
            act_idx = info["action"].squeeze(-1).long()
            info["act_emb"] = self.action_encoder(act_idx)
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        return preds
