from mha.qk_fine import diagonaled_mm_qk_fine_ltr
from mha.qk_coarse import diagonaled_mm_qk_coarse_ltr
from mha.cv_fine import diagonaled_mm_cv_fine_ltr
from mha.cv_coarse import diagonaled_mm_cv_coarse_ltr

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfMHAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_pdrop = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.m = config.m
        self.scale_attn = config.scale_attn
        self.block_size = config.block_size
        self.attn = MHA(self.block_size, self.m, self.n_embd//self.n_head, config.p, self.attn_pdrop, self.scale_attn, config.downsampling)
        
    def forward(self, x):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.contiguous().view(B*self.n_head, T, C // self.n_head)
        q = q.contiguous().view(B*self.n_head, T, C // self.n_head)
        v = v.contiguous().view(B*self.n_head, T, C // self.n_head)
        q = self.attn(q,k,v)
        q = q.view(B, self.n_head, T, C // self.n_head)
        q = q.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(q))


class MHA(torch.nn.Module):
    def __init__(self, n:int, m:int, d:int, p:int, attn_pdrop, scale_attn, downsampling):

        self.n, self.m, self.d, self.p, self.attn_pdrop, self.scale_attn, self.downsampling = n, m, d, p, attn_pdrop, scale_attn, downsampling
        super(MHA, self).__init__()
        self.L = int(math.log2(int(n/m))) - 1
        if self.downsampling == "groupedconv":
            print("MLA using grouped conv on k,v")
            conv_k = []
            conv_v = []
            for l in range(self.L):
                w = m*(2**l)
                conv_k.append(nn.Conv1d(d, p*d, w, stride=w, groups=d))
                conv_v.append(nn.Conv1d(d, p*d, w, stride=w, groups=d))
            self.conv_k = nn.ModuleList(conv_k)
            self.conv_v = nn.ModuleList(conv_v)
        elif self.downsampling == "conv":
            print("MLA using conv on k,v")
            conv_k = []
            conv_v = []
            for l in range(self.L):
                w = m*(2**l)
                conv_k.append(nn.Conv1d(d, p*d, w, stride=w))
                conv_v.append(nn.Conv1d(d, p*d, w, stride=w))
            self.conv_k = nn.ModuleList(conv_k)
            self.conv_v = nn.ModuleList(conv_v)
        elif self.downsampling == "avgpool":
            print("MLA using avgpool on k,v")
            pool_k = []
            pool_v = []
            for l in range(self.L):
                w = m*(2**l)
                pool_k.append(nn.AvgPool1d(w//p, stride=w//p))
                pool_v.append(nn.AvgPool1d(w//p, stride=w//p))
            self.pool_k = nn.ModuleList(pool_k)
            self.pool_v = nn.ModuleList(pool_v) 
        return

    def stage1(self, Q, Kls):
        L, m, attn_pdrop, scale_attn, p= self.L, self.m, self.attn_pdrop, self.scale_attn, self.p
        B, n, d = Q.shape
        device = Q.device
        As = []
        Cs = []
        exp_sums = torch.zeros((B, n, L+1)).to(device)
        max_attn_weights = torch.zeros((B, n, L+1)).to(device)
        for l in range(L+1):
            Kl = Kls[l]
            A = diagonaled_mm_qk_fine_ltr(Q, Kl, m) if l==0 else diagonaled_mm_qk_coarse_ltr(Q, Kl, self.p)
            As.append(A)
        for l in range(L+1):
            max_attn_weights[:, :, l] = torch.max(As[l], dim=2, keepdim=False)[0]
        max_attn_weights_all_levels = torch.max(max_attn_weights, dim=2, keepdim=True)[0]
        for l in range(L+1):
            As[l] = As[l] - max_attn_weights_all_levels
        for l in range(L+1):
            exp_sums[:, :, l] = torch.sum(torch.exp(As[l]), dim=2)
            if scale_attn and l>=1:
                exp_sums[:, :, l] *= (m*(2**(l-1))) // p
        exp_sum = torch.sum(exp_sums, dim=2, keepdim=True)
        for l in range(L+1):
            scale =  (m*(2**(l-1))) // p if (l!=0 and scale_attn) else 1
            Cs.append(scale * torch.exp(As[l]) / exp_sum)
            Cs[l] = torch.nn.functional.dropout(Cs[l], p=attn_pdrop, training = self.training)
        return Cs



    def convolute(self, K, V):
        Kl, Vl = K, V
        Kls = [Kl]
        Vls = [Vl]
        b,n,d = K.shape
        p = self.p
        for l in range(self.L):
            w = self.m*(2**l)
            nl = p * (n//w)
            if self.downsampling == "groupedconv" or self.downsampling == "conv":
                Kl = self.conv_k[l](K.transpose(1,2))
                assert(Kl.shape == (b, d*p, n//w)), f"conv{l} out shape{Kl.shape} neq {(b, d*p, n//w)}."
                Kl = Kl.view((b, d, p, n//w))
                Kl = Kl.transpose(1,3)
                Kl = Kl.reshape((b, nl, d)).contiguous()
                Vl = self.conv_v[l](V.transpose(1,2))
                assert(Vl.shape == (b, d*p, n//w)), f"conv{l} out shape{Vl.shape} neq {(b, d*p, n//w)}."
                Vl = Vl.view((b, d, p, n//w))
                Vl = Vl.transpose(1,3)
                Vl = Vl.reshape((b, nl, d)).contiguous()
                Kls.append(Kl)
                Vls.append(Vl)         
            elif self.downsampling == "avgpool":
                Kl = self.pool_k[l](K.transpose(1,2))
                assert(Kl.shape == (b, d, (n//w)*p)), f"conv{l} out shape{Kl.shape} neq {(b, d*p, n//w)}."
                Kl = Kl.transpose(1, 2).contiguous()
                Vl = self.pool_v[l](V.transpose(1,2))
                assert(Vl.shape == (b, d, (n//w)*p)), f"conv{l} out shape{Vl.shape} neq {(b, d*p, n//w)}."
                Vl = Vl.transpose(1, 2).contiguous()
                Kls.append(Kl)
                Vls.append(Vl)
        return Kls, Vls


    def stage2(self, Cs, Vls):
        m = self.m
        for l in range(self.L + 1):
            if l==0:
                result = diagonaled_mm_cv_fine_ltr(Cs[l], Vls[l], m)   
            else:
                result_l = diagonaled_mm_cv_coarse_ltr(Cs[l], Vls[l], self.p)
                result += result_l
        return result

    def forward(self, Q, K, V):
        Q = Q / math.sqrt(self.d)
        Kls, Vls = self.convolute(K, V)
        Cs = self.stage1( Q, Kls)
        result = self.stage2(Cs, Vls)
        return result








