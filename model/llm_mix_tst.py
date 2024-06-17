from typing import Tuple, Optional
import struct
import inspect
import math
import numpy as np
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from .layers import RevIN, make_linear_layer
from .layers import TransformerBlock, RMSNorm, precompute_freqs_cis, PatchChannelHead, PatchMixChannelHead

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    multiple_of: int = 16  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 200
    dropout: float = 0.0
    device: str = 'cuda'
    
    # for time decoder
    n_channel: int = 5
    patch_len: int = 10
    stride: int = 5
    n_predict_patch: int = 2
    n_min_patch: int = 2 # 头部的Patch保留两个不做预测
    use_time_feature: bool = False
    time_feature_dim: int = 5
    price_channel:int = 0
    
class LLMMixTST(nn.Module):
    """
    基于语言模型预测的时间序列，使用patch编码，对整个patch进行预测
    注意1：不同时间类型的model最好加上起始虚拟patch or 增加time feature
    注意2：只关注某些维度预测时可以调节不同变量的loss权重
    注意3：为了降低预测的难度，计算loss时可以计算整个patch的变量均值和方差作为loss
    """
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        # self.loss_weight = torch.Tensor([1.0] * params.n_channel).to(params.device)

        assert params.max_seq_len % params.stride == 0 
         # patch number等同于 sequence length，这里取固定size
        self.patch_num = int((params.max_seq_len - params.patch_len) / params.stride + 1)
        # 在最后padding一下，保证真实的最后的预测loss能够计算出来
        self.revin_layer = RevIN(params.n_channel, affine=False, subtract_last=False)
        self.patch_proj = make_linear_layer(params.patch_len, params.dim)
        if params.use_time_feature:
            self.time_proj = make_linear_layer(params.time_feature_dim * params.patch_len, params.dim)

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.head_channel = nn.Sequential(nn.Dropout(params.dropout), 
                                  PatchChannelHead(individual=True, 
                                                   n_channel=params.n_channel, 
                                                   h_dim=params.dim,
                                                   out_dim=params.dim))
        self.head_mix =  PatchMixChannelHead(params.n_channel, params.dim, 
                                          params.n_predict_patch * params.patch_len)
        
        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor, time_feature: torch.Tensor = None) -> torch.Tensor:
        if time_feature is None:
            assert not self.params.use_time_feature
        # z: input shape [bs x seq_len x nvar]
        # time_feature: input shape [bs x seq_len x time_dim]
        # norm
        z = self.revin_layer(z, 'norm')
        z = z.permute(0,2,1)
        # z: [bs x nvars x patch_num x patch_len]
        z = z.unfold(dimension=-1, size=self.params.patch_len, step=self.params.stride)
        # time feature convert
        # z: [bs x nvars x patch_num x hidden_size]
        enc_in = self.patch_proj(z)  
        if time_feature is not None and self.params.use_time_feature:
            # [bs x patch_num x patch_len x time_dim]
            time_feature = time_feature.unfold(dimension=-2, size=self.params.patch_len, step=self.params.stride)
            time_feature = torch.flatten(time_feature, start_dim=2)
            time_feature = self.time_proj(time_feature)
            time_feature = torch.unsqueeze(time_feature, dim=1)
            enc_in = time_feature + enc_in
        # full_enc_int: [bs * nvars x patch_num x hidden_size]
        h = torch.reshape(enc_in, (enc_in.shape[0]*enc_in.shape[1],
                                                    enc_in.shape[2],
                                                    enc_in.shape[3]))
        h = self.dropout(h)
        seqlen = h.shape[1]
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        # enc_out: [bs x nvars x patch_num x hidden_size]
        # 根据T，预测T+1，T+2的patch
        # 根据T，T+1，预测 T+2，T+3的patch
        enc_out = torch.reshape(h, (-1, self.params.n_channel, h.shape[-2], h.shape[-1]))
        # [bs x nvar x patch_num x predict_patch_num * patch_length]
        output = self.head_channel(enc_out)
        output = torch.reshape(output, (-1, self.params.n_channel, 
                                self.patch_num * self.params.dim))
        # denorm
        output = output.permute(0,2,1)
        output = self.revin_layer(output, 'denorm')
        output = output.permute(0,2,1)

        output = torch.reshape(output, (-1, self.params.n_channel, self.patch_num, self.params.dim))

        # shape = batch x patch_num x dim
        output = self.head_mix(output)
        output = torch.reshape(output, (-1, self.patch_num, 
                                        self.params.n_predict_patch,
                                        self.params.patch_len))
        return output

    def pretrain_loss(self, input_seq, seq_length, predict_out):
        input_seq = input_seq[:, :, self.params.price_channel]
        input_seq = input_seq.unfold(dimension=-1, size=self.params.patch_len, step=self.params.stride)
        # batch patch num
        patch_real_num = ((seq_length - self.params.patch_len) / self.params.stride + 1)
        patch_real_num = patch_real_num.unsqueeze(1)
        patch_mask = torch.arange(self.patch_num).unsqueeze(0).to(seq_length.device)
        patch_mask = patch_mask < patch_real_num
        patch_mask = patch_mask[:, :, None]
        batch_loss = []
        for idx in range(0, self.params.n_predict_patch):
            pred = predict_out[:, :-(1+idx), idx, :]
            # input = batch x patch_num x patch_len
            target = input_seq[:, idx+1:, :]
            mask = patch_mask[:, idx+1:, :]
            loss = F.mse_loss(pred, target, reduction="none")
            # shape = [bs x var x patch_num x patch_length]
            loss = (loss * mask).sum()
            batch_loss.append(loss)
        total_loss = torch.sum(torch.vstack(batch_loss), dim=0)
        all_loss = torch.sum(total_loss)
        return all_loss, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

if __name__ == "__main__":
    args = ModelArgs()
    args.n_channel = 2
    args.max_seq_len = 20
    batch_size = 3
    model = LLMMixTST(args).to('cuda')
    
    inputs = torch.randn(batch_size, args.max_seq_len, args.n_channel)
    seq_len = torch.ones(batch_size) * 5
    predict_out = model(inputs.to('cuda'))
    loss = model.pretrain_loss(inputs.to('cuda'), seq_len.to('cuda'), predict_out)
