# -*- coding: utf-8 -*-
"""
Neural MCTS测试脚本
"""

import time
import numpy as np
from typing import Dict, List, Tuple
import json

# 导入必要的模块
from integrated_ai import IntegratedGomokuAI
from neural_mcts import NeuralMCTSAdapter, AlphaZeroStyleNetwork, NeuralEvaluatorNetworkAdapter
from neural_evaluator import NeuralNetworkEvaluator, NeuralConfig

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

def test_neural_mcts_basic():
    """
    基本的Neural MCTS功能测试
    """
    print("开始Neural MCTS基本功能测试...")
    
    try:
        # 创建Neural MCTS系统
        print("1. 创建Neural MCTS系统...")
        network = AlphaZeroStyleNetwork()
        mcts_adapter = NeuralMCTSAdapter(
            neural_network=network,
            mcts_simulations=100,
            c_puct=1.0
        )
        print("   [OK] Neural MCTS系统创建成功")
        
        # 测试基本移动
        print("2. 测试基本移动功能...")
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[7][7] = BLACK
        
        start_time = time.time()
        result = mcts_adapter.get_best_move(board, WHITE)
        elapsed_time = time.time() - start_time
        
        if hasattr(result, 'move') and result.move:
            print(f"   [OK] 获取移动成功: {result.move}, 用时: {elapsed_time:.2f}s")
            print(f"   分数: {result.score}, 搜索节点: {result.nodes_searched}")
        else:
            print("   [ERROR] 获取移动失败")
            return False
        
        # 测试评估功能
        print("3. 测试棋盘评估功能...")
        score = mcts_adapter.evaluate_board(board, WHITE)
        print(f"   [OK] 棋盘评估分数: {score}")
        
        # 测试统计信息
        print("4. 测试统计信息...")
        stats = mcts_adapter.get_statistics()
        print(f"   [OK] 统计信息获取成功")
        
        print("Neural MCTS基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"Neural MCTS基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_ai():
    """
    测试集成AI系统中的Neural MCTS
    """
    print("开始集成AI系统测试...")
    
    try:
        # 创建Neural MCTS AI
        print("1. 创建Neural MCTS AI系统...")
        ai_system = IntegratedGomokuAI(
            ai_difficulty="neural_mcts",
            time_limit=3.0
        )
        print("   [OK] Neural MCTS AI系统创建成功")
        
        # 获取AI信息
        print("2. 获取AI系统信息...")
        ai_info = ai_system.get_ai_info()
        print(f"   引擎类型: {ai_info.get('engine_type', 'Unknown')}")
        print(f"   难度: {ai_info.get('difficulty', 'Unknown')}")
        print("   主要特性:")
        for feature in ai_info.get('features', [])[:3]:
            print(f"     - {feature}")
        
        # 测试移动
        print("3. 测试AI移动...")
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[7][7] = BLACK
        board[7][8] = WHITE
        
        start_time = time.time()
        move = ai_system.get_ai_move(board, BLACK)
        elapsed_time = time.time() - start_time
        
        if move:
            print(f"   [OK] AI移动成功: {move}, 用时: {elapsed_time:.2f}s")
        else:
            print("   [ERROR] AI移动失败")
            return False
        
        print("集成AI系统测试通过！")
        return True
        
    except Exception as e:
        print(f"集成AI系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison():
    """
    简化的性能比较测试
    """
    print("开始性能比较测试...")
    
    # 创建测试棋盘
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board[7][7] = BLACK
    board[7][8] = WHITE
    board[8][7] = BLACK
    
    engines = [
        ("传统Minimax", "medium", "minimax"),
        ("Neural MCTS", "neural_mcts", "neural_mcts")
    ]
    
    results = []
    
    for name, difficulty, engine_type in engines:
        print(f"测试引擎: {name}")
        try:
            ai_system = IntegratedGomokuAI(
                ai_difficulty=difficulty,
                time_limit=2.0,
                engine_type=engine_type
            )
            
            start_time = time.time()
            move = ai_system.get_ai_move(board, WHITE)
            elapsed_time = time.time() - start_time
            
            results.append({
                'name': name,
                'move': move,
                'time': elapsed_time,
                'success': move is not None
            })
            
            status = "[OK]" if move else "[ERROR]"
            print(f"  {status} 移动: {move}, 时间: {elapsed_time:.2f}s")
            
        except Exception as e:
            print(f"  [ERROR] 错误: {e}")
            results.append({
                'name': name,
                'move': None,
                'time': float('inf'),
                'success': False,
                'error': str(e)
            })
    
    # 显示结果
    print("比较结果:")
    print("-" * 40)
    for result in results:
        status = "[OK]" if result['success'] else "[ERROR]"
        print(f"{status} {result['name']}: {result['time']:.2f}s")
    
    return results

def main():
    """
    主测试函数
    """
    print("Neural MCTS 测试程序")
    print("=" * 40)
    
    # 基本功能测试
    print("阶段1: 基本功能测试")
    if not test_neural_mcts_basic():
        print("基本功能测试失败，跳过后续测试")
        return
    
    # 集成AI测试
    print("\\n阶段2: 集成AI系统测试")
    if not test_integrated_ai():
        print("集成AI测试失败")
    
    # 性能比较测试
    print("\\n阶段3: 性能比较测试")
    try:
        results = test_comparison()
        print("\\n所有测试完成！")
        
        # 保存结果
        with open("neural_mcts_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print("测试结果已保存到: neural_mcts_test_results.json")
        
    except Exception as e:
        print(f"性能比较测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()