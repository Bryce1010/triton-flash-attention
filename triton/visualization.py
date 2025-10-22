"""
Flash Attention 可视化工具

这个模块提供了可视化Flash Attention计算过程的工具，
帮助理解分块计算和在线softmax算法。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


def visualize_attention_matrix(attention_scores: torch.Tensor, 
                             title: str = "Attention Matrix",
                             save_path: Optional[str] = None,
                             show_colorbar: bool = True):
    """
    可视化注意力矩阵
    
    Args:
        attention_scores: 注意力分数矩阵 [seq_len, seq_len]
        title: 图表标题
        save_path: 保存路径（可选）
        show_colorbar: 是否显示颜色条
    """
    plt.figure(figsize=(10, 8))
    
    # 转换为numpy数组
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.detach().cpu().numpy()
    
    # 创建热力图
    sns.heatmap(attention_scores, 
                cmap='Blues', 
                cbar=show_colorbar,
                square=True,
                xticklabels=False,
                yticklabels=False)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Key Position', fontsize=12)
    plt.ylabel('Query Position', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_causal_mask(seq_len: int, 
                         title: str = "Causal Attention Mask"):
    """
    可视化因果注意力掩码
    
    Args:
        seq_len: 序列长度
        title: 图表标题
    """
    # 创建下三角掩码
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(mask.numpy(), 
                cmap='RdYlBu_r', 
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False,
                vmin=0, vmax=1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Key Position', fontsize=12)
    plt.ylabel('Query Position', fontsize=12)
    
    # 添加说明
    plt.text(seq_len//2, -seq_len//10, 
             'White: Can Attend, Blue: Masked', 
             ha='center', fontsize=10)
    
    plt.show()


def visualize_block_computation(seq_len: int, 
                              block_size_q: int, 
                              block_size_kv: int,
                              title: str = "Flash Attention Block Computation"):
    """
    可视化Flash Attention的分块计算过程
    
    Args:
        seq_len: 序列长度
        block_size_q: Q块大小
        block_size_kv: K/V块大小
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Q矩阵分块
    q_blocks = torch.zeros(seq_len, seq_len)
    for i in range(0, seq_len, block_size_q):
        end_i = min(i + block_size_q, seq_len)
        q_blocks[i:end_i, :] = (i // block_size_q) + 1
    
    sns.heatmap(q_blocks.numpy(), ax=axes[0], cmap='Set3', 
                cbar=True, square=True, xticklabels=False, yticklabels=False)
    axes[0].set_title('Q Matrix Blocks', fontweight='bold')
    axes[0].set_xlabel('Head Dimension')
    axes[0].set_ylabel('Sequence Position')
    
    # 2. K/V矩阵分块
    kv_blocks = torch.zeros(seq_len, seq_len)
    for j in range(0, seq_len, block_size_kv):
        end_j = min(j + block_size_kv, seq_len)
        kv_blocks[:, j:end_j] = (j // block_size_kv) + 1
    
    sns.heatmap(kv_blocks.numpy(), ax=axes[1], cmap='Set2', 
                cbar=True, square=True, xticklabels=False, yticklabels=False)
    axes[1].set_title('K/V Matrix Blocks', fontweight='bold')
    axes[1].set_xlabel('Sequence Position')
    axes[1].set_ylabel('Head Dimension')
    
    # 3. 注意力计算模式
    attention_pattern = torch.zeros(seq_len, seq_len)
    for i in range(0, seq_len, block_size_q):
        for j in range(0, seq_len, block_size_kv):
            end_i = min(i + block_size_q, seq_len)
            end_j = min(j + block_size_kv, seq_len)
            block_id = (i // block_size_q) * (seq_len // block_size_kv) + (j // block_size_kv)
            attention_pattern[i:end_i, j:end_j] = block_id + 1
    
    sns.heatmap(attention_pattern.numpy(), ax=axes[2], cmap='viridis', 
                cbar=True, square=True, xticklabels=False, yticklabels=False)
    axes[2].set_title('Attention Computation Order', fontweight='bold')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def compare_attention_implementations(Q: torch.Tensor, 
                                    K: torch.Tensor, 
                                    V: torch.Tensor,
                                    causal: bool = False,
                                    head_idx: int = 0,
                                    batch_idx: int = 0):
    """
    比较标准注意力和Flash Attention的结果
    
    Args:
        Q, K, V: 输入张量
        causal: 是否使用因果掩码
        head_idx: 要可视化的头索引
        batch_idx: 要可视化的批次索引
    """
    seq_len = Q.shape[2]
    head_dim = Q.shape[3]
    softmax_scale = 1 / (head_dim ** 0.5)
    
    # 标准实现
    print("🔍 计算标准注意力实现...")
    P_std = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    
    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
        P_std[:, :, mask == 0] = float('-inf')
    
    P_std = torch.softmax(P_std.float(), dim=-1).to(V.dtype)  # 确保数据类型匹配
    O_std = torch.matmul(P_std, V)
    
    # Flash Attention实现
    print("⚡ 计算Flash Attention实现...")
    from flash_attention_debug import TritonAttentionDebug
    O_flash = TritonAttentionDebug.apply(Q, K, V, causal, softmax_scale)
    
    # 提取单个头的结果进行可视化
    P_vis = P_std[batch_idx, head_idx].detach().cpu()
    O_std_vis = O_std[batch_idx, head_idx].detach().cpu()
    O_flash_vis = O_flash[batch_idx, head_idx].detach().cpu()
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：注意力矩阵和输出
    sns.heatmap(P_vis.numpy(), ax=axes[0, 0], cmap='Blues', 
                cbar=True, square=True, xticklabels=False, yticklabels=False)
    axes[0, 0].set_title('Attention Weights', fontweight='bold')
    
    sns.heatmap(O_std_vis.numpy(), ax=axes[0, 1], cmap='viridis', 
                cbar=True, xticklabels=False, yticklabels=False)
    axes[0, 1].set_title('Standard Attention Output', fontweight='bold')
    
    sns.heatmap(O_flash_vis.numpy(), ax=axes[0, 2], cmap='viridis', 
                cbar=True, xticklabels=False, yticklabels=False)
    axes[0, 2].set_title('Flash Attention Output', fontweight='bold')
    
    # 第二行：差异分析
    diff = torch.abs(O_std_vis - O_flash_vis)
    
    axes[1, 0].hist(P_vis.flatten().numpy(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_title('Attention Weights Distribution')
    axes[1, 0].set_xlabel('Attention Score')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].plot(O_std_vis.mean(dim=1).numpy(), label='Standard', linewidth=2)
    axes[1, 1].plot(O_flash_vis.mean(dim=1).numpy(), label='Flash', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Output Mean by Position')
    axes[1, 1].set_xlabel('Sequence Position')
    axes[1, 1].set_ylabel('Mean Output Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    sns.heatmap(diff.numpy(), ax=axes[1, 2], cmap='Reds', 
                cbar=True, xticklabels=False, yticklabels=False)
    axes[1, 2].set_title(f'Absolute Difference\n(Max: {diff.max():.2e})')
    
    plt.suptitle(f'Attention Comparison ({"Causal" if causal else "Non-Causal"})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n📊 结果统计:")
    print(f"   最大差异: {diff.max():.2e}")
    print(f"   平均差异: {diff.mean():.2e}")
    print(f"   相对误差: {(diff.mean() / O_std_vis.abs().mean()):.2e}")


def create_learning_visualization():
    """
    创建Flash Attention学习可视化
    """
    print("🎨 创建Flash Attention学习可视化...")
    
    # 1. 因果掩码可视化
    print("\n1️⃣ 因果注意力掩码")
    visualize_causal_mask(16, "Causal Attention Mask (16x16)")
    
    # 2. 分块计算可视化
    print("\n2️⃣ 分块计算模式")
    visualize_block_computation(64, 16, 16, "Flash Attention Block Computation (64x64)")
    
    # 3. 小规模实际比较
    print("\n3️⃣ 实际计算比较")
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 32, 64
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # 非因果注意力比较
    print("\n   📈 非因果注意力比较")
    compare_attention_implementations(Q, K, V, causal=False, head_idx=0)
    
    # 因果注意力比较
    print("\n   📈 因果注意力比较")
    compare_attention_implementations(Q, K, V, causal=True, head_idx=0)


if __name__ == "__main__":
    # 检查依赖
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        create_learning_visualization()
    except ImportError as e:
        print(f"❌ 缺少可视化依赖: {e}")
        print("请安装: pip install matplotlib seaborn")
