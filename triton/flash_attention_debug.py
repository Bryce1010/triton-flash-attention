import torch
import triton
import triton.language as tl
import numpy as np


def print_debug(msg, *args):
    """Debug打印函数"""
    print(f"[DEBUG] {msg}", *args)


def print_tensor_info(name, tensor):
    """打印张量信息"""
    if tensor is not None:
        print(f"[TENSOR] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        if tensor.numel() < 20:  # 只有小张量才打印具体值
            print(f"         values={tensor.flatten()}")
        else:
            print(f"         min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")


@triton.jit
def _attn_fwd_inner_debug(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    """带调试信息的内部前向传播函数"""
    
    # 确定处理范围
    if STAGE == 1:
        # 对角线左边的块（非因果注意力的一部分）
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # 对角线上的块（需要掩码的过渡块）
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # 非因果注意力（处理整个序列）
        lo, hi = 0, SEQ_LEN

    # 移动K和V指针到正确位置
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # 主循环：遍历K、V块
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # 1. 计算注意力分数 QK^T
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        # 2. 应用缩放和掩码
        if STAGE == 2:
            # 因果掩码：只能看到当前位置及之前的位置
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # 非因果情况：直接应用缩放
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # 3. 计算softmax（在线算法）
        P_block = tl.math.exp(QK_block)  # exp(QK - max)
        l_ij = tl.sum(P_block, 1)        # 当前块的行和

        # 4. 更新运行统计量（Flash Attention的核心）
        alpha = tl.math.exp(m_i - m_ij)  # 修正因子
        l_i = l_i * alpha + l_ij         # 更新运行和
        
        # 5. 计算输出并累加
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        
        # 关键步骤：O_new = P × V + O_old × alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        # 6. 更新最大值
        m_i = m_ij

        # 7. 移动到下一个K、V块
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd_debug(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    """带调试信息的前向传播主函数"""
    
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # 获取当前处理的块索引
    block_index_q = tl.program_id(0)  # 处理第几个Q块
    index_batch_head = tl.program_id(1)  # 处理第几个batch×head
    
    # 解析batch和head索引
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # 计算内存偏移量
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    # 创建块指针（高效的内存访问模式）
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    # K矩阵需要转置，所以strides是反的
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # 计算偏移量
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # 初始化累加器（Flash Attention的关键变量）
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")  # 运行最大值
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0           # 运行和
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)   # 输出累加器

    # 加载Q块（在整个计算过程中保持在SRAM中）
    Q_block = tl.load(Q_block_ptr)

    # 根据STAGE执行不同的计算策略
    if STAGE == 1 or STAGE == 3:
        # 非因果注意力 或 因果注意力的左半部分
        O_block, l_i, m_i = _attn_fwd_inner_debug(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
            block_index_q, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            4 - STAGE, offs_q, offs_kv, SEQ_LEN,
        )

    if STAGE == 3:
        # 因果注意力的对角线块（需要掩码）
        O_block, l_i, m_i = _attn_fwd_inner_debug(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
            block_index_q, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            2, offs_q, offs_kv, SEQ_LEN,
        )

    # 最终处理：计算logsumexp并归一化
    m_i += tl.math.log(l_i)  # 计算logsumexp（反向传播需要）
    O_block = O_block / l_i[:, None]  # 最终归一化

    # 存储结果
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


class TritonAttentionDebug(torch.autograd.Function):
    """带调试信息的Triton Attention类"""

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        print_debug("=== Flash Attention 前向传播开始 ===")
        print_tensor_info("输入Q", Q)
        print_tensor_info("输入K", K)
        print_tensor_info("输入V", V)
        print_debug(f"causal={causal}, softmax_scale={softmax_scale:.6f}")
        
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        print_debug(f"矩阵维度: BATCH_SIZE={BATCH_SIZE}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}")

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1
        print_debug(f"执行模式: STAGE={stage} ({'因果注意力' if causal else '非因果注意力'})")

        # 计算GPU网格大小
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # M存储logsumexp值（反向传播需要）
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        print_debug(f"创建logsumexp矩阵M: {M.shape}")

        print_debug("开始执行Triton kernel...")
        _attn_fwd_debug[grid](
            Q=Q, K=K, V=V, softmax_scale=softmax_scale, M=M, O=O,
            stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2), stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0), stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2), stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0), stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2), stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0), stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2), stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0], NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2], HEAD_DIM=HEAD_DIM_K, STAGE=stage,
        )

        print_tensor_info("输出O", O)
        print_tensor_info("logsumexp M", M)
        print_debug("=== Flash Attention 前向传播完成 ===\n")

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        print_debug("=== Flash Attention 反向传播开始 ===")
        Q, K, V, O, M = ctx.saved_tensors
        print_tensor_info("梯度dO", dO)

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        print_debug(f"反向传播参数: MICRO={BLOCK_SIZE_MICRO}, MACRO={BLOCK_SIZE_MACRO}")

        # 预处理：计算D = rowsum(dO ⊙ O)
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)
        print_debug("步骤1: 计算D矩阵（预处理）...")

        # 使用原始模块的反向传播函数
        import importlib.util
        import os
        
        # 动态导入flash_attention模块
        flash_attention_path = os.path.join(os.path.dirname(__file__), 'flash_attention.py')
        spec = importlib.util.spec_from_file_location("flash_attention", flash_attention_path)
        flash_attention = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flash_attention)
        
        _attn_bwd_preprocess = flash_attention._attn_bwd_preprocess
        _attn_bwd_dk_dv = flash_attention._attn_bwd_dk_dv
        _attn_bwd_dq = flash_attention._attn_bwd_dq
        _attn_bwd_preprocess[preprocess_grid](
            O=O, dO=dO, D=D, SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO, HEAD_DIM=ctx.HEAD_DIM,
        )

        print_tensor_info("D矩阵", D)

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1

        print_debug("步骤2: 计算dK和dV...")
        _attn_bwd_dk_dv[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale, dO=dO,
            dQ=dQ, dK=dK, dV=dV, M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1),
            stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO, BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM, STAGE=stage,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        print_debug("步骤3: 计算dQ...")
        _attn_bwd_dq[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale, dO=dO,
            dQ=dQ, dK=dK, dV=dV, M=M, D=D,
            stride_batch=Q.stride(0), stride_head=Q.stride(1),
            stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO, BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM, STAGE=stage,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        print_tensor_info("梯度dQ", dQ)
        print_tensor_info("梯度dK", dK)
        print_tensor_info("梯度dV", dV)
        print_debug("=== Flash Attention 反向传播完成 ===\n")

        return dQ, dK, dV, None, None


def test_debug_attention():
    """测试debug版本的Flash Attention"""
    print("🚀 开始测试Flash Attention (Debug版本)")
    print("=" * 60)
    
    # 使用较小的参数便于观察
    BATCH_SIZE = 2
    NUM_HEADS = 4
    SEQ_LEN = 256  # 较小的序列长度
    HEAD_DIM = 64
    dtype = torch.float16

    print_debug(f"测试参数: B={BATCH_SIZE}, H={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}")

    # 创建输入张量
    Q = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
    K = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
    V = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    print("\n" + "="*60)
    print("🔍 测试1: 非因果注意力")
    print("="*60)
    
    # 测试非因果注意力
    causal = False
    
    # 参考实现
    print_debug("计算参考实现...")
    P_ref = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    P_ref = torch.softmax(P_ref.float(), dim=-1).to(V.dtype)
    ref_O = torch.matmul(P_ref, V)
    print_tensor_info("参考输出", ref_O)

    # Flash Attention实现
    print_debug("计算Flash Attention实现...")
    tri_O = TritonAttentionDebug.apply(Q, K, V, causal, softmax_scale).half()

    # 比较结果
    diff = torch.abs(ref_O - tri_O)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print_debug(f"结果比较: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✅ 非因果注意力测试通过!")
    else:
        print("❌ 非因果注意力测试失败!")

    print("\n" + "="*60)
    print("🔍 测试2: 因果注意力")
    print("="*60)
    
    # 测试因果注意力
    causal = True
    
    # 参考实现
    print_debug("计算因果注意力参考实现...")
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P_ref = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    P_ref[:, :, MASK == 0] = float("-inf")
    P_ref = torch.softmax(P_ref.float(), dim=-1).to(V.dtype)
    ref_O = torch.matmul(P_ref, V)
    print_tensor_info("因果注意力参考输出", ref_O)

    # Flash Attention实现
    print_debug("计算因果Flash Attention实现...")
    tri_O = TritonAttentionDebug.apply(Q, K, V, causal, softmax_scale).half()

    # 比较结果
    diff = torch.abs(ref_O - tri_O)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print_debug(f"因果注意力结果比较: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✅ 因果注意力测试通过!")
    else:
        print("❌ 因果注意力测试失败!")

    print("\n" + "="*60)
    print("🎯 Flash Attention 核心概念总结")
    print("="*60)
    print("1. 🧩 分块计算: 将大矩阵分成小块，逐块处理")
    print("2. 📊 在线softmax: 使用运行最大值m_i和运行和l_i")
    print("3. 🔄 修正因子α: exp(m_old - m_new)用于更新之前的结果")
    print("4. 💾 内存效率: 避免存储完整的注意力矩阵")
    print("5. ⚡ GPU优化: 利用SRAM减少内存访问")


if __name__ == "__main__":
    test_debug_attention()
