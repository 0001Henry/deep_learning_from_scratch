import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, head_num):
        super().__init__()
        self.embed_size = embed_size   
        self.head_num = head_num
        self.head_dim = embed_size // head_num
        
        assert(
            self.head_dim * head_num == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        # Strictly speaking, d_q == d_k != d_q, here, and not necessarily == embed_size
        self.Q = nn.Linear(embed_size,embed_size)
        self.K = nn.Linear(embed_size,embed_size)
        self.V = nn.Linear(embed_size,embed_size)
        
        self.fc_out = nn.Linear(embed_size,embed_size)
        
    def forward(self,q,k,v,mask=None):
        batch_size = q.shape[0]
        
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]
        
        q = self.Q(q).reshape(batch_size, q_len, self.head_num, self.head_dim)
        k = self.K(k).reshape(batch_size, k_len, self.head_num, self.head_dim)
        v = self.V(v).reshape(batch_size, v_len, self.head_num, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        
        if mask is not None: # which mask ?  
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        # k_len == v_len here
        out = torch.einsum("nhqv,nvhd->nqhd", [attention, v]).reshape(
            batch_size, q_len, self.head_num * self.head_dim
        )
        
        out = self.fc_out(out)
        
        # return out, attention
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, head_num, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, head_num)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        attention = self.attention(q, k, v, mask)
        x = self.norm1(attention + q)   # Add & Norm
        x = self.dropout(x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)   # Add & Norm
        out = self.dropout(out)
        return out
    

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        head_num,
        device,
        dropout=0.1,
        forward_expansion=4,
        max_length=100,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embed = nn.Embedding(src_vocab_size, embed_size)
        self.position_embed = nn.Embedding(max_length, embed_size) # different
        
        self.layers = nn.ModuleList(
            [
                EncoderBlock(embed_size, head_num, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        out = self.dropout(
            self.word_embed(x) + self.position_embed(positions)
        )
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, head_num, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, head_num)
        self.norm = nn.LayerNorm(embed_size)
        self.sub_block = EncoderBlock(
            embed_size, head_num, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, v, k, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        q = self.dropout(self.norm(attention + x))
        out = self.sub_block(q, k, v, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        head_num,
        device,
        dropout=0.1,
        forward_expansion=4,
        max_length=100,
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embed = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embed = nn.Embedding(max_length, embed_size) # different
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, head_num, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        out = self.dropout(
            self.word_embed(x) + self.position_embed(positions)
        )
        
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)
            
        out = self.fc_out(out)
        return out
    
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            dropout,
            forward_expansion,
            max_length,
        )
        
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            dropout,
            forward_expansion,
            max_length,
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
               
    def _make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def _make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self._make_src_mask(src)
        trg_mask = self._make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2], [1, 5, 6, 2, 4, 7, 6]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out)
    print(out.shape)

    
