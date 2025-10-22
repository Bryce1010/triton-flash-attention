#!/usr/bin/env python3
"""
Flash Attention Debug 运行脚本

这个脚本会运行带有详细调试信息的Flash Attention实现，
帮助你理解算法的每个步骤。
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
triton_dir = os.path.join(current_dir, 'triton')
sys.path.insert(0, current_dir)
sys.path.insert(0, triton_dir)

import torch

# 导入我们的模块
try:
    from flash_attention_debug import test_debug_attention
    DEBUG_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入debug模块: {e}")
    DEBUG_AVAILABLE = False

# 尝试导入可视化模块
try:
    from visualization import create_learning_visualization
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 可视化模块不可用: {e}")
    VISUALIZATION_AVAILABLE = False

def main():
    print("🎓 Flash Attention 学习工具")
    print("=" * 60)
    print("这个工具会展示Flash Attention的详细执行过程")
    print("包括:")
    print("  • 输入输出张量的形状和数值范围")
    print("  • 每个计算步骤的说明")
    print("  • 与标准实现的结果比较")
    print("  • 核心概念的总结")
    print("=" * 60)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ 错误: 需要CUDA支持才能运行Triton")
        print("请确保:")
        print("  1. 安装了支持CUDA的PyTorch")
        print("  2. 系统有可用的GPU")
        return
    
    print(f"✅ CUDA设备: {torch.cuda.get_device_name()}")
    print(f"✅ PyTorch版本: {torch.__version__}")
    
    # 检查模块可用性
    if not DEBUG_AVAILABLE:
        print("❌ Debug模块不可用，无法继续")
        return
    
    try:
        # 询问用户是否要运行可视化
        if VISUALIZATION_AVAILABLE:
            print("\n🤔 你想要运行哪个部分?")
            print("1. 仅运行调试版本 (快速)")
            print("2. 运行调试版本 + 可视化 (需要matplotlib)")
            print("3. 仅运行可视化")
            
            choice = input("\n请选择 (1/2/3, 默认1): ").strip() or "1"
        else:
            print("\n⚠️  可视化功能不可用 (缺少matplotlib/seaborn)")
            print("如需可视化，请运行: pip install matplotlib seaborn")
            choice = "1"
        
        if choice in ["1", "2"]:
            # 运行debug测试
            test_debug_attention()
        
        if choice in ["2", "3"] and VISUALIZATION_AVAILABLE:
            print("\n🎨 开始可视化...")
            create_learning_visualization()
        
        print("\n" + "🎉" * 20)
        print("恭喜! Flash Attention学习完成!")
        print("🎉" * 20)
        
        print("\n📚 进一步学习建议:")
        print("1. 尝试修改SEQ_LEN参数，观察性能差异")
        print("2. 比较因果和非因果注意力的计算过程")
        print("3. 研究不同BLOCK_SIZE对内存使用的影响")
        print("4. 阅读Flash Attention论文了解理论背景")
        print("5. 运行可视化工具理解分块计算过程")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("\n🔧 可能的解决方案:")
        print("1. 检查Triton是否正确安装: pip install triton")
        print("2. 确保GPU内存足够")
        print("3. 尝试减小测试参数（SEQ_LEN等）")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
