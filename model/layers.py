import numpy as np

import torch
from torch import Tensor
from torch import nn

class MovingAverage(nn.Module):
    "Moving average block to highlight the trend of time series"

    def __init__(self,
         kernel_size:int,  # the size of the window
         ):
        super().__init__()
        padding_left = (kernel_size - 1) // 2
        padding_right = kernel_size - padding_left - 1
        self.padding = torch.nn.ReplicationPad1d((padding_left, padding_right))
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x:Tensor):
        """ 
        Args:
            x: torch.Tensor shape: [bs x seq_len x features]
        """
        return self.avg(self.padding(x))


class SeriesDecomposition(nn.Module):
    "Series decomposition block"

    def __init__(self,
         kernel_size:int,  # the size of the window
         ):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x:Tensor):
        """ Args:
            x: torch.Tensor shape: [bs x seq_len x features]
        """
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)  # type: ignore

    @staticmethod
    def _init_weight(out: torch.Tensor) -> torch.Tensor:
        """
        Features are not interleaved. The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        # set early to avoid an error in pytorch-1.8+
        out.requires_grad = False

        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(  # type: ignore
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x var x seqlen x ...]."""
        seq_len = input_ids_shape[2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)
    
class RevIN(nn.Module):
    """对batch的输入重新进行归一化"""
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

    def forward(self, x, mode: str):
        # input shape: [b, seq, var]
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

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
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x): 
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
    def __repr__(self): 
        if self.contiguous: return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else: return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"

    
def make_linear_layer(dim_in, dim_out):
    lin = nn.Linear(dim_in, dim_out)
    torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
    torch.nn.init.zeros_(lin.bias)
    return lin

class FlattenHead(nn.Module):
    def __init__(self, individual: bool, n_vars, nf, h, c_out, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        self.c_out = c_out
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, h*c_out))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, h*c_out)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x hidden_size x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x hidden_size * patch_num]
                z = self.linears[i](z)                    # z: [bs x h]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x h]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x