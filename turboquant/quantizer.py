import torch
import torch.nn as nn
import math
from .packing import pack_indices, unpack_indices

class _TurboQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, centroids):
        distances = torch.abs(y.unsqueeze(-1) - centroids)
        idx = torch.argmin(distances, dim=-1)
        y_quantized = centroids[idx]
        ctx.save_for_backward(y, y_quantized)
        return y_quantized, idx

    @staticmethod
    def backward(ctx, grad_y_quantized, grad_idx):
        return grad_y_quantized, None


class TurboQuantMSELayer(nn.Module):
    def __init__(self, d: int, b: int):
        super().__init__()
        self.d = d
        self.b = b
        self.num_centroids = 2 ** b
        
        rand_mat = torch.randn(d, d)
        q, _ = torch.linalg.qr(rand_mat)
        self.register_buffer('Pi', q)
        
        optimal_centroids = torch.linspace(-1.0, 1.0, self.num_centroids)
        self.register_buffer('centroids', optimal_centroids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.Pi.T)
        y_quantized, _ = _TurboQuantSTE.apply(y, self.centroids)
        x_tilde = torch.matmul(y_quantized, self.Pi)
        return x_tilde

    def encode_inference(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = torch.matmul(x, self.Pi.T)
            distances = torch.abs(y.unsqueeze(-1) - self.centroids)
            idx = torch.argmin(distances, dim=-1)
            return pack_indices(idx, self.b)

    def decode_inference(self, packed_idx: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            idx = unpack_indices(packed_idx, self.b, self.d)
            y_quantized = self.centroids[idx]
            return torch.matmul(y_quantized, self.Pi)


class TurboQuantProdLayer(nn.Module):
    def __init__(self, d: int, b: int):
        super().__init__()
        if b < 2:
            raise ValueError("Bit-width b must be >= 2 for TurboQuantProd.")
        
        self.d = d
        self.b = b
        self.mse_layer = TurboQuantMSELayer(d, b - 1)
        
        self.register_buffer('S', torch.randn(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tilde_mse = self.mse_layer(x)
        r = x - x_tilde_mse
        
        Sr = torch.matmul(r, self.S.T)
        qjl_signs = Sr + (torch.sign(Sr) - Sr).detach()  
        
        gamma = torch.norm(r, p=2, dim=-1, keepdim=True)
        scaling_factor = math.sqrt(math.pi / 2) / self.d
        x_tilde_qjl = scaling_factor * gamma * torch.matmul(qjl_signs, self.S)
        
        return x_tilde_mse + x_tilde_qjl
