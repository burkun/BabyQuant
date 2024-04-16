import torch
from torch import nn
from .layers import RevIN, make_linear_layer, SinusoidalPositionalEmbedding,  FlattenHead, Transpose

### https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb

class PatchTSTModel(nn.Module):
    """
    Module implementing the PatchTST model for forecasting as described in
    https://arxiv.org/abs/2211.14730

    Parameters
    ----------
    c_in
        Number of channel
    predict_len
        Number of time points to predict.
    context_len
        Number of time steps prior to prediction time that the model.
    patch_len
        one patch length 
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    norm_type:
        layer, batch
    train_type:
        pretrain, supervise
    """

    def __init__(
        self,
        c_in: int,
        context_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_encoder_layers: int,
        norm_type: str,
        train_type: str,
        predict_len: int = 0,
        pre_c_out: int = 0,
        individual: bool = True
    ) -> None:
        super().__init__()
        if train_type != "pretrain":
            assert predict_len > 0
        assert context_len > 0
        assert c_in > 0
        self.c_in = c_in 
        self.predict_len = predict_len
        self.context_len = context_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.padding_patch = padding_patch
        self.norm_type = norm_type
        self.train_type = train_type
        self.individual = individual

        # patch number等同于 sequence length，这里取固定size
        self.patch_num = int((context_len - patch_len) / stride + 1)
        # TODO know how
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1
        self.revin_layer = RevIN(c_in, affine=False, subtract_last=False)
        # 所有时间变量input输入共享一个全连接，TODO，后续可以分开使用AutoEncoder训练
        self.patch_proj = make_linear_layer(patch_len, d_model)
        # 位置编码
        self.positional_encoding = SinusoidalPositionalEmbedding(
            self.patch_num, d_model
        )

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )

        if self.norm_type == "batch":
            encoder_norm = nn.Sequential(nn.Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        if self.train_type == "pretrain":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(d_model * self.patch_num, self.c_in, 1))
        else:
            self.head = FlattenHead(self.individual, self.c_in, 
                                    d_model * self.patch_num, 
                                    predict_len, 
                                    pre_c_out, 
                                    head_dropout=dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, z):
        # z = input shape [bs x nvars x seq_len]
        # norm
        z = z.permute(0,2,1)
        z = self.revin_layer(z, 'norm')
        z = z.permute(0,2,1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        # z: [bs x nvars x patch_num x patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z: [bs x nvars x patch_num x hidden_size]
        enc_in = self.patch_proj(z)
        embed_pos = self.positional_encoding(enc_in.size())
        # embedding 输入的dropout
        full_enc_int = self.proj_dropout(enc_in + embed_pos)
        # full_enc_int: [bs * nvars x patch_num x hidden_size]
        full_enc_int = torch.reshape(full_enc_int, shape = 
                                     (full_enc_int.shape[0] * full_enc_int.shape[1],
                                      full_enc_int.shape[2],
                                      full_enc_int.shape[3]))
        # enc_out: [bs * nvars x patch_num x hidden_size]
        enc_out = self.encoder(full_enc_int)
        # enc_out: [bs x nvars x hidden_size x patch_num]
        enc_out = torch.reshape(enc_out, (-1, self.c_in, enc_out.shape[-2], enc_out.shape[-1]))
        # if head is pretrain [bs x nvars x 1]; if head is supervise [bs x nvar x predict_day * c_out]
        z = self.head(enc_out)
        # denorm
        z = z.permute(0,2,1)
        z = self.revin_layer(z, 'denorm')
        z = z.permute(0,2,1)
        return z