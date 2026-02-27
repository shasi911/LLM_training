import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    For each pair of dimensions (2i, 2i+1) and position m, applies:
        x'_{2i}   = x_{2i}  * cos(m·θ_i) - x_{2i+1} * sin(m·θ_i)
        x'_{2i+1} = x_{2i}  * sin(m·θ_i) + x_{2i+1} * cos(m·θ_i)
    where θ_i = 1 / theta^(2i / d_k).

    cos/sin tables are pre-computed up to max_seq_len and stored as buffers
    (not learnable parameters).
    """

    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super().__init__()
        # θ_i = 1 / theta^(2i/d_k),  i = 0 … d_k/2 - 1
        i = torch.arange(0, d_k // 2, dtype=torch.float32)
        freqs = 1.0 / (theta ** (2 * i / d_k))          # (d_k/2,)

        # Pre-compute angles for every position 0 … max_seq_len-1
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)            # (max_seq_len, d_k/2)

        self.register_buffer("cos_cache", torch.cos(angles))  # (max_seq_len, d_k/2)
        self.register_buffer("sin_cache", torch.sin(angles))  # (max_seq_len, d_k/2)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x:               (..., seq_len, d_k)
        # token_positions: (..., seq_len)  -- integer positions
        cos = self.cos_cache[token_positions]   # (..., seq_len, d_k/2)
        sin = self.sin_cache[token_positions]   # (..., seq_len, d_k/2)

        # Split into even / odd dimension pairs
        x1 = x[..., 0::2]   # (..., seq_len, d_k/2)
        x2 = x[..., 1::2]   # (..., seq_len, d_k/2)

        # Apply rotation and interleave back to (..., seq_len, d_k)
        out = torch.stack([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax: subtract max before exponentiating."""
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Average cross-entropy loss over a batch.

    inputs:  (batch_size, vocab_size)  -- unnormalized logits
    targets: (batch_size,)             -- integer class indices

    Uses the log-sum-exp trick for numerical stability:
        CE = -logit[target] + log( sum( exp(logits) ) )
           = -logit[target] + max + log( sum( exp(logits - max) ) )
    """
    # Subtract max per row for numerical stability
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))   # (batch,)
    # Gather the shifted logit at the target index for each example
    target_logits = shifted[torch.arange(inputs.shape[0]), targets]  # (batch,)
    loss = -target_logits + log_sum_exp                        # (batch,)
    return loss.mean()


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Scaled dot-product attention.

    scores = Q @ K^T / sqrt(d_k)
    if mask provided: set masked-out positions (mask==False) to -inf
    output = softmax(scores, dim=-1) @ V

    Q: (..., queries, d_k)
    K: (..., keys,    d_k)
    V: (..., keys,    d_v)
    mask: (..., queries, keys) bool  -- True = attend, False = mask out
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ V


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention (no positional encoding)."""

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape

        # Project then split into (batch, num_heads, seq_len, d_head)
        def split_heads(proj):
            return proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.q_proj)
        K = split_heads(self.k_proj)
        V = split_heads(self.v_proj)

        # Causal mask: position i can only attend to positions j <= i
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Merge heads back: (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(out)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Causal multi-head self-attention with RoPE applied to Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(d_k=self.d_head, theta=theta, max_seq_len=max_seq_len)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        batch, seq_len, _ = x.shape

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # Project then split into (batch, num_heads, seq_len, d_head)
        def split_heads(proj):
            return proj(x).view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.q_proj)
        K = split_heads(self.k_proj)
        V = split_heads(self.v_proj)

        # Apply RoPE to every head (broadcasts over batch and num_heads dims)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # Causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(out)


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.

    token_ids  →  Embedding  →  N × TransformerBlock  →  RMSNorm  →  Linear  →  logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch, seq_len)
        x = self.token_embeddings(token_ids)          # (batch, seq_len, d_model)

        seq_len = token_ids.shape[-1]
        token_positions = torch.arange(seq_len, device=token_ids.device)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)                        # (batch, seq_len, vocab_size)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    Uses RoPE inside attention and SwiGLU for the FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class SwiGLU(nn.Module):
    """
    Position-wise feed-forward network with SwiGLU activation.

    FFN(x) = W2( SiLU(W1·x) ⊙ W3·x )

    W1, W3: (d_model -> d_ff)  -- gate and value projections
    W2:     (d_ff   -> d_model) -- down projection
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable per-element gain, initialised to 1 (identity at start).
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Keep input dtype; compute norm in float32 for numerical stability.
        in_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Weight matrix: (num_embeddings, embedding_dim)
        # Each row is the embedding vector for one token.
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        # Simple index into the embedding matrix -- no matrix multiply needed.
        return self.embedding[token_ids]


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store W as (out_features, in_features) for memory-ordering reasons.
        # forward computes x @ W.T, equivalent to the standard linear transform.
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.W, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W.T


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    Update rule for each parameter θ with gradient g at step t:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t²
        θ_t = θ_{t-1} * (1 - lr * λ)
              - lr / (1 - β1^t) * m_t / (√(v_t / (1 - β2^t)) + ε)

    Weight decay is applied directly to the parameters (decoupled from the
    gradient-based update), as in Loshchilov & Hutter (2019).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy state initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-correction factors
                bias_corr1 = 1 - beta1 ** t
                bias_corr2 = 1 - beta2 ** t

                # Decoupled weight decay
                p.mul_(1 - lr * weight_decay)

                # Adam gradient step (bias-corrected)
                step_size = lr / bias_corr1
                denom = (v.sqrt() / (bias_corr2 ** 0.5)).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

        return loss


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients of all parameters so their combined L2 norm is at most max_l2_norm.

    Computes the global norm across all parameter gradients, then rescales every
    gradient by min(1, max_l2_norm / global_norm) so the total norm never exceeds
    max_l2_norm. Parameters with no gradient are skipped.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Three regions:
      1. Linear warmup:  0 <= it < warmup_iters
            lr = max_lr * it / warmup_iters
      2. Cosine decay:   warmup_iters <= it <= cosine_cycle_iters
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
            where progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
      3. Post-cycle:     it > cosine_cycle_iters
            lr = min_lr
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
        1 + math.cos(math.pi * progress)
    )


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of language modelling examples from a token dataset.

    Randomly draws `batch_size` start positions uniformly from
    [0, len(dataset) - context_length), then returns:
        x: tokens  at positions [start, start + context_length)
        y: tokens  at positions [start + 1, start + context_length + 1)  (next-token targets)

    Both tensors have shape (batch_size, context_length) and dtype torch.long.
    """
    n = len(dataset)
    starts = np.random.randint(0, n - context_length, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )
