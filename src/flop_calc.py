
# ✅ FLOP Calculation for Matrix Multiplication
def compute_matmul_flops(m, n, p):
    """
    Computes FLOPs for a matrix multiplication: (m x n) @ (n x p).
    Formula: m * p * (2n - 1)
    """
    return m * p * (2 * n - 1)

# ✅ FLOP Calculation for Self-Attention
def compute_self_attention_flops(seq_len, d_model, num_heads):
    """
    Computes FLOPs for Multi-Head Self-Attention layer.
    """
    head_dim = d_model // num_heads
    qkv_proj = 3 * compute_matmul_flops(seq_len, d_model, d_model)  # Q, K, V projections
    attention_scores = compute_matmul_flops(seq_len, head_dim, seq_len) * num_heads  # Attention dot product
    attention_output = compute_matmul_flops(seq_len, seq_len, head_dim) * num_heads  # Attention applied to values
    output_proj = compute_matmul_flops(seq_len, d_model, d_model)  # Final projection

    return qkv_proj + attention_scores + attention_output + output_proj

# ✅ FLOP Calculation for MLP (SwiGLU)
def compute_mlp_flops(seq_len, d_model, d_ffn):
    """
    Computes FLOPs for MLP block with SwiGLU activation.
    """
    gate_proj = compute_matmul_flops(seq_len, d_model, d_ffn)  # First linear projection
    activation = seq_len * d_ffn  # SwiGLU activation
    down_proj = compute_matmul_flops(seq_len, d_ffn, d_model)  # Second linear projection

    return gate_proj + activation + down_proj

# ✅ FLOP Calculation for LayerNorm
def compute_layernorm_flops(seq_len, d_model):
    """
    Computes FLOPs for LayerNorm.
    """
    return seq_len * d_model  # Element-wise ops

# ✅ Compute FLOPs for LoRA-modified Q and V projections
def compute_lora_flops(seq_len, d_model, num_heads, lora_rank):
    """
    Computes FLOPs for LoRA-modified Q and V projections.
    """
    head_dim = d_model // num_heads
    return 2 * num_heads * (2 * head_dim * lora_rank) * seq_len

# ✅ Compute Total FLOPs for One Forward Pass
def compute_forward_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank):
    """
    Computes total FLOPs for a forward pass, including LoRA modifications.
    """
    total_flops = 0
    for _ in range(num_layers):
        total_flops += compute_self_attention_flops(seq_len, d_model, num_heads)
        total_flops += compute_mlp_flops(seq_len, d_model, d_ffn)
        total_flops += compute_layernorm_flops(seq_len, d_model) * 2  # 2x LayerNorm per layer
        total_flops += compute_lora_flops(seq_len, d_model, num_heads, lora_rank)  # LoRA contribution
    return total_flops

# ✅ Compute Total FLOPs for Training (Forward + Backward)
def compute_training_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank, num_steps):
    """
    Computes total FLOPs for training with LoRA.
    """
    forward_flops = compute_forward_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank)
    return forward_flops * 3 * num_steps  # Backprop is ~2x forward pass

# ✅ Compute Total FLOPs for Inference (Forward Only)
def compute_inference_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank, num_samples):
    """
    Computes total FLOPs for inference.
    """
    return compute_forward_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank) * num_samples

# ✅ Track FLOPs During Training
def track_training_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank, num_steps):
    train_flops = compute_training_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank, num_steps)
    print(f"Total Training FLOPs: {train_flops:.2E}")

# ✅ Track FLOPs During Inference
def track_inference_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank, num_samples):
    eval_flops = compute_inference_flops(seq_len, d_model, num_heads, d_ffn, num_layers, lora_rank, num_samples)
    print(f"Total Inference FLOPs: {eval_flops:.2E}")