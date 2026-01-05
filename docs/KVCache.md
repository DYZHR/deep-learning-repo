# 理论

## 什么是KV Cache

KV Cache，就是在模型推理时，把已经计算过的token对应的k和v保存起来。

当处理下一个token时，把过去token的k和v，分别和当前token计算出的k和v拼接起来，用作当前的k和v，避免重复计算过去token的k和v。



## 为什么需要

在模型推理阶段，对一个固定的token而言，K和V的映射后张量在不同token的推理时，是不变的。

比如处理token1时，先经过Linear层k_proj和v_proj的映射生成k1和v1，然后经过一系列步骤生成token2。当生成token2时，token1还会经过Linear层k_proj和v_proj生成k1'和v1'，而k1'=k1，同样v1=v1'。所以这里过去的token的Linear层计算是不必要的。

所以可以把已经计算出的K和V保存起来，用于后面的token推理。



## 为什么只用于推理阶段

因为训练阶段，Linear层的参数会变化，token1经过Linear层先后生成的k1、v1和k1'、v1'是不等的，而且一句话只训练一次，所以保存已经计算出的K和V没有意义。



## 为什么Q不能缓存

推理阶段，每次生成新的token，其实只需要1个token计算q。

因为token计算出的q，对应是用于计算token的下一个token。而推理只需要最新一个token生成的下一个token，所以过去token生成的q没有用，缓存起来只是增加显存占用和计算开销。



# 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np


class RotaryEmbedding(nn.Module):
    """
    修复维度匹配问题：
    1. 扩展cos/sin的维度，匹配x的batch/num_heads维度
    2. 截取cos/sin的前head_dim//2维度，与x1/x2匹配
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim  # 新增：保存head_dim，方便forward使用
        # 预计算旋转角度：theta = 10000^(-2(i-1)/d)，i为维度索引（仅前half维度）
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)  # [head_dim//2]

        # 预计算位置索引：0,1,2,...,max_seq_len-1
        pos_idx = torch.arange(max_seq_len, dtype=torch.float32)
        self.register_buffer("pos_idx", pos_idx)  # [max_seq_len]

        # 预计算旋转矩阵的角度：m * theta → [max_seq_len, head_dim//2]
        m_theta = torch.outer(self.pos_idx, self.theta)
        # 无需repeat_interleave！后续在forward中直接广播到完整head_dim
        self.register_buffer("cos_base", torch.cos(m_theta))  # [max_seq_len, head_dim//2]
        self.register_buffer("sin_base", torch.sin(m_theta))  # [max_seq_len, head_dim//2]

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Args:
            x: 输入张量，shape [batch_size, num_heads, seq_len, head_dim]
            start_pos: 当前序列的起始位置
        Returns:
            编码后的x: shape与输入一致
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim, f"x的head_dim({head_dim})与RoPE的head_dim({self.head_dim})不匹配"

        # 1. 截取当前序列对应的cos/sin（shape: [seq_len, head_dim//2]）
        cos = self.cos_base[start_pos:start_pos + seq_len]
        sin = self.sin_base[start_pos:start_pos + seq_len]

        # 2. 扩展维度，匹配x的batch/num_heads维度（shape: [1, 1, seq_len, head_dim//2]）
        cos = cos.unsqueeze(0).unsqueeze(0)  # 新增：扩展前两个维度
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 3. 拆分x为前后两半（各head_dim//2）
        x1 = x[..., :head_dim // 2]  # [batch, num_heads, seq_len, head_dim//2]
        x2 = x[..., head_dim // 2:]  # [batch, num_heads, seq_len, head_dim//2]

        # 4. 核心旋转操作（此时cos/sin与x1/x2维度完全匹配）
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 5. 拼接回完整head_dim
        rotated_x = torch.cat([rotated_x1, rotated_x2], dim=-1)  # [batch, num_heads, seq_len, head_dim]

        return rotated_x


class KVCacheAttention(nn.Module):
    """
    缓存历史KV，Q每次重新算
    """
    def __init__(
            self,
            hidden_dim: int = 4096,
            num_heads: int = 32,
            max_seq_len: int = 2048,
            dropout: float = 0.0):  # 可不用，留个接口
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = hidden_dim // num_heads

        assert num_heads * self.head_dim == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)


    """
    前向传播：核心处理KV Cache的拼接与复用
    Args:
        x: 输入序列，推理时单Token输入为[batch, 1, hidden_dim]，Prompt阶段为[batch, seq_len, hidden_dim]
        kv_cache: 历史KV缓存，shape均为 [batch, num_heads, past_seq_len, head_dim]
        start_pos: 起始位置（Prompt阶段为0，增量生成时为已生成序列长度）
        use_cache: 控制是否使用KV缓存
    Returns:
        output: 注意力输出，shape [batch_size, seq_len, hidden_dim]
        new_kv_cache: 更新后的KV缓存 (new_k, new_v)
    """
    def forward(
            self,
            x: torch.Tensor,
            kv_cache: tuple = None,
            start_pos: int = 0,
            use_cache: bool = True
    ):
        # ——————————————————1. QKV线性投影和维度变换——————————————————
        batch_size, seq_len, hidden_dim = x.size()
        # [b, n, d]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [b, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # —————————————————2. 旋转编码加入相对位置信息————————————————
        q = self.rotary_emb(q, start_pos)
        k = self.rotary_emb(k, start_pos)

        # —————————————————3. KV Cache拼接————————————————————————
        if use_cache:
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        # —————————————————4. 计算注意力分数——————————————————
        # [b, num_heads, seq_len, seq_len + past_seq_len]
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if use_cache or seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, k.size(2), device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, -float('inf'))

        # ——————————————————5. 注意力权重和输出————————————————
        # [b, num_heads, seq_len, seq_len + past_seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)
        # [b, num_heads, seq_len, head_dim]
        attn_out = torch.matmul(attn_weights, v)

        # ——————————————————6. 还原形状和输出——————————————————
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, -1)

        output = self.out_proj(attn_out)
        return output, new_kv_cache


```

