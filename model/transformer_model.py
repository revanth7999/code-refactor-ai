
import torch.nn as nn

class TransformerCodeGen(nn.Module):
    def __init__(self, vocab_size, embed_size=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src).permute(1, 0, 2)
        tgt_emb = self.embedding(tgt).permute(1, 0, 2)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc(output).permute(1, 0, 2)
