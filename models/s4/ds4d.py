"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd

class DS4DKernel(nn.Module):
    def __init__(self, d_model, N=64, lr=None):
        super().__init__()
        H = d_model
        self.H = H
        self.N = N

        # 학습 파라미터
        self.C = nn.Parameter(torch.randn(H, N))
        self.B = nn.Parameter(torch.randn(N))
        self.A = nn.Parameter(torch.eye(N))  # 기본 A

    def forward(self, du):
        """
        du: (B, H, L) 입력
        return: K (H, L)
        """
        B, H, L = du.shape
        A = self.A
        C = self.C
        Bvec = self.B

        # deltaA 생성: proj_u로부터 계산 가능 (여기선 간단히 du를 선형 투영)
        deltaA = torch.einsum('bhl,nh->bnl', du, torch.randn(H, self.N, device=du.device))

        # 초기 A 상태
        A_t = A.unsqueeze(0).repeat(B, 1, 1)  # (B, N, N)

        Ks = []
        for t in range(L):
            A_t = A_t + torch.diag(deltaA[:, :, t])  # ΔA 적용
            K_t = torch.einsum('bn,n->b', (C, (A_t @ Bvec)))
            Ks.append(K_t)

        K = torch.stack(Ks, dim=-1)  # (B, H, L)
        return K


    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class DS4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.D = nn.Parameter(torch.randn(self.h))

        self.kernel = DS4DKernel(self.h, N=self.n)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_linear = nn.Conv1d(self.h, self.h, kernel_size=1)

    def forward(self, u):
        """u: (B, H, L)"""
        du = u[:, :, 1:] - u[:, :, :-1]   # du 계산
        du = F.pad(du, (1, 0))            # 첫 step padding

        K = self.kernel(du)               # einsum 기반 K 생성

        # convolution
        k_f = torch.fft.rfft(K, n=2*u.size(-1))
        u_f = torch.fft.rfft(u, n=2*u.size(-1))
        y = torch.fft.irfft(u_f * k_f, n=2*u.size(-1))[..., :u.size(-1)]

        # skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        return y, None
