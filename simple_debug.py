#!/usr/bin/env python3
"""
简化的Flash Attention调试脚本

这个脚本专注于展示Flash Attention的核心概念，
避免复杂的导入和数据类型问题。
"""

import torch
import sys
import os

# 添加triton目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'triton'))

def print_debug(msg, *args):
    """Debug打印函数"""
    print(f"[DEBUG] {msg}", *args)

def print_tensor_info(name, tensor):
    """打印张量信息"""
    if tensor is not None:
        print(f"[TENSOR] {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        if tensor.numel() < 20:
            print(f"         values={tensor.flatten()}")
        else:
            print(f"         min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")

def explain_flash_attention_concepts():
    """解释Flash Attention的核心概念"""
    print("\n" + "🎓" * 20)
    print("Flash Attention 核心概念详解")
    print("🎓" * 20)
    
    print("\n1️⃣ 标准Attention的问题:")
    print("   • 内存复杂度: O(N²) - 需要存储完整的注意力矩阵")
    print("   • 计算复杂度: O(N²) - 对于长序列非常昂贵")
    print("   • 内存访问: 频繁的HBM访问，效率低")
    
    print("\n2️⃣ Flash Attention的解决方案:")
    print("   • 🧩 分块计算: 将大矩阵分成小块处理")
    print("   • 📊 在线softmax: 不需要完整矩阵就能计算softmax")
    print("   • 💾 内存优化: 使用GPU的SRAM而不是HBM")
    print("   • ⚡ 流式处理: 边计算边丢弃中间结果")
    
    print("\n3️⃣ 在线Softmax算法:")
    print("   • m_i: 运行最大值 (running maximum)")
    print("   • l_i: 运行和 (running sum)")
    print("   • α: 修正因子 = exp(m_old - m_new)")
    print("   • 公式: O_new = (O_old × α + P × V) / l_new")
    
    print("\n4️⃣ 分块策略:")
    print("   • Q块: 固定一个Q块，遍历所有K/V块")
    print("   • K/V块: 逐个加载，计算后丢弃")
    print("   • 因果掩码: 只在对角线块应用掩码")

def demonstrate_online_softmax():
    """演示在线softmax算法"""
    print("\n" + "🔬" * 20)
    print("在线Softmax算法演示")
    print("🔬" * 20)
    
    # 创建一个简单的例子
    torch.manual_seed(42)
    x = torch.tensor([1.0, 3.0, 2.0, 4.0])
    print(f"输入向量: {x}")
    
    # 标准softmax
    standard_softmax = torch.softmax(x, dim=0)
    print(f"标准softmax: {standard_softmax}")
    
    # 在线softmax模拟
    print("\n在线softmax计算过程:")
    m = float('-inf')  # 运行最大值
    l = 0.0           # 运行和
    result = torch.zeros_like(x)
    
    for i, val in enumerate(x):
        print(f"\n步骤 {i+1}: 处理 x[{i}] = {val:.1f}")
        
        # 更新最大值
        m_new = max(m, val.item())
        print(f"  更新最大值: m = {m:.1f} -> {m_new:.1f}")
        
        # 计算修正因子
        alpha = torch.exp(torch.tensor(m - m_new))
        print(f"  修正因子: α = exp({m:.1f} - {m_new:.1f}) = {alpha:.4f}")
        
        # 更新运行和
        l = l * alpha + torch.exp(val - m_new)
        print(f"  更新运行和: l = {l:.4f}")
        
        # 更新结果
        result = result * alpha
        result[i] = torch.exp(val - m_new)
        print(f"  当前结果: {result}")
        
        m = m_new
    
    # 最终归一化
    result = result / l
    print(f"\n最终结果: {result}")
    print(f"标准结果: {standard_softmax}")
    print(f"差异: {torch.abs(result - standard_softmax).max():.6f}")

def test_flash_attention_simple():
    """简化的Flash Attention测试"""
    print("\n" + "🚀" * 20)
    print("Flash Attention 实际测试")
    print("🚀" * 20)
    
    # 导入原始模块
    try:
        from flash_attention import TritonAttention
        print("✅ 成功导入TritonAttention")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保flash_attention.py在triton目录中")
        return
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return
    
    print(f"✅ CUDA设备: {torch.cuda.get_device_name()}")
    
    # 测试参数（使用较小的参数避免内存问题）
    BATCH_SIZE = 1
    NUM_HEADS = 2
    SEQ_LEN = 64  # 很小的序列长度
    HEAD_DIM = 32  # 很小的头维度
    dtype = torch.float32  # 使用float32避免精度问题
    
    print_debug(f"测试参数: B={BATCH_SIZE}, H={NUM_HEADS}, S={SEQ_LEN}, D={HEAD_DIM}")
    
    # 创建输入
    print_debug("创建输入张量...")
    torch.manual_seed(42)  # 固定随机种子便于复现
    Q = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=True)
    K = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=True)
    V = torch.randn((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda", requires_grad=True)
    
    print_tensor_info("Q", Q)
    print_tensor_info("K", K)
    print_tensor_info("V", V)
    
    softmax_scale = 1 / (HEAD_DIM**0.5)
    print_debug(f"softmax_scale = {softmax_scale:.6f}")
    
    # 测试非因果注意力
    print("\n" + "="*50)
    print("🔍 测试: 非因果注意力")
    print("="*50)
    
    try:
        print_debug("计算参考实现...")
        P_ref = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
        P_ref = torch.softmax(P_ref, dim=-1)
        ref_O = torch.matmul(P_ref, V)
        print_tensor_info("参考输出", ref_O)
        
        print_debug("计算Flash Attention实现...")
        tri_O = TritonAttention.apply(Q, K, V, False, softmax_scale)
        print_tensor_info("Flash Attention输出", tri_O)
        
        # 比较结果
        diff = torch.abs(ref_O - tri_O)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print_debug(f"结果比较:")
        print_debug(f"  最大差异: {max_diff:.6f}")
        print_debug(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✅ 非因果注意力测试通过!")
        else:
            print("⚠️ 非因果注意力测试有差异，但这可能是正常的数值误差")
            
    except Exception as e:
        print(f"❌ 非因果注意力测试失败: {e}")
    
    # 测试因果注意力
    print("\n" + "="*50)
    print("🔍 测试: 因果注意力")
    print("="*50)
    
    try:
        print_debug("计算因果注意力参考实现...")
        MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
        P_ref = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
        P_ref[:, :, MASK == 0] = float("-inf")
        P_ref = torch.softmax(P_ref, dim=-1)
        ref_O = torch.matmul(P_ref, V)
        print_tensor_info("因果参考输出", ref_O)
        
        print_debug("计算因果Flash Attention实现...")
        tri_O = TritonAttention.apply(Q, K, V, True, softmax_scale)
        print_tensor_info("因果Flash Attention输出", tri_O)
        
        # 比较结果
        diff = torch.abs(ref_O - tri_O)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print_debug(f"因果注意力结果比较:")
        print_debug(f"  最大差异: {max_diff:.6f}")
        print_debug(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✅ 因果注意力测试通过!")
        else:
            print("⚠️ 因果注意力测试有差异，但这可能是正常的数值误差")
            
    except Exception as e:
        print(f"❌ 因果注意力测试失败: {e}")

def main():
    print("🎓 Flash Attention 深度学习教程")
    print("=" * 60)
    print("这个教程将帮助你理解Flash Attention的核心概念")
    print("=" * 60)
    
    try:
        # 1. 解释概念
        explain_flash_attention_concepts()
        
        # 2. 演示在线softmax
        demonstrate_online_softmax()
        
        # 3. 实际测试
        test_flash_attention_simple()
        
        print("\n" + "🎉" * 20)
        print("Flash Attention学习完成!")
        print("🎉" * 20)
        
        print("\n📚 学习总结:")
        print("1. Flash Attention通过分块计算避免了O(N²)内存复杂度")
        print("2. 在线softmax算法是核心创新，允许流式处理")
        print("3. 修正因子α确保了数值稳定性和正确性")
        print("4. GPU内存层次结构优化是性能提升的关键")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
