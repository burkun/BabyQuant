import torch
import torch.nn as nn

from types import Optional

import model_utils
import attentions as attn

# code from https://github.com/ts-kim/RevIN, with minor modifications

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    

class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        q_len,
        hidden_size,
        n_heads,
        d_k=None,
        d_v=None,
        linear_hidden_size=256,
        store_attn=False,
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        res_attention=False,
        pre_norm=False,
    ):
        super().__init__()
        assert (
            not hidden_size % n_heads
        ), f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = attn.MultiheadAttention(
            hidden_size,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), 
                nn.BatchNorm1d(hidden_size), 
                Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(hidden_size)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden_size, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(linear_hidden_size, hidden_size, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(hidden_size)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):  # -> Tuple[torch.Tensor, Any]:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
    
class TSTEncoder(nn.Module):
    def __init__(
        self,
        q_len,
        hidden_size,
        n_heads,
        d_k=None,
        d_v=None,
        linear_hidden_size=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    q_len,
                    hidden_size,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    linear_hidden_size=linear_hidden_size,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            return output
        

class TSTiEncoder(nn.Module):  # i means channel-independent
    """
    patch num：patch数量
    patch len: patch长度
    max_seq_len：最大长度
    """
    def __init__(
        self,
        c_in,
        patch_num,
        patch_len,
        max_seq_len=1024,
        n_layers=3,
        hidden_size=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        linear_hidden_size=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        store_attn=False,
        key_padding_mask="auto",
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
    ):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding，qlen = patch数量
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, hidden_size
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = model_utils.positional_encoding(pe, learn_pe, q_len, hidden_size)

        # Residual dropout, 残差后的dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            q_len,
            hidden_size,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x) -> torch.Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x hidden_size]

        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x hidden_size]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x hidden_size]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x hidden_size]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x hidden_size]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x hidden_size x patch_num]

        return z
