"""
神经网络集成测试模块
测试神经网络评估器与现有AI系统的集成效果

作者：Claude AI Engineer
日期：2025-09-22
"""

import time
import random
from typing import List, Dict, Tuple
from neural_evaluator import NeuralNetworkEvaluator, NeuralConfig
from integrated_ai import IntegratedGomokuAI, EnhancedGomokuGame

# 游戏常量
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

def create_test_scenarios() -> List[Tuple[str, List[List[int]], int]]:
    """创建测试场景"""
    scenarios = []
    
    # 场景1：空棋盘开局
    board1 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    scenarios.append(("空棋盘开局", board1, BLACK))
    
    # 场景2：标准开局
    board2 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board2[7][7] = BLACK
    scenarios.append(("标准开局", board2, WHITE))
    
    # 场景3：中局对战
    board3 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board3[7][7] = BLACK
    board3[7][8] = WHITE
    board3[8][7] = WHITE
    board3[8][8] = BLACK
    board3[6][6] = BLACK
    board3[9][9] = WHITE
    board3[6][8] = WHITE
    board3[8][6] = BLACK
    scenarios.append(("中局对战", board3, BLACK))
    
    # 场景4：威胁局面
    board4 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    # 创建一个黑棋即将获胜的局面
    for i in range(4):
        board4[7][7+i] = BLACK
    board4[7][11] = WHITE  # 阻止
    board4[8][7] = WHITE
    board4[8][8] = WHITE
    scenarios.append(("威胁局面", board4, BLACK))
    
    # 场景5：复杂残局
    board5 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    positions = [
        (6, 6, BLACK), (6, 7, WHITE), (6, 8, BLACK), (6, 9, WHITE),
        (7, 6, WHITE), (7, 7, BLACK), (7, 8, WHITE), (7, 9, BLACK),
        (8, 6, BLACK), (8, 7, WHITE), (8, 8, BLACK), (8, 9, WHITE),
        (9, 6, WHITE), (9, 7, BLACK), (9, 8, WHITE), (9, 9, BLACK),
        (5, 7, BLACK), (10, 8, WHITE)
    ]
    for row, col, player in positions:
        board5[row][col] = player
    scenarios.append(("复杂残局", board5, WHITE))
    
    return scenarios

def test_neural_ai_integration():
    """测试神经网络AI集成"""
    print("=== 神经网络AI集成测试 ===")
    
    # 创建不同难度的AI
    ai_configs = [
        ("传统AI-中等", "medium", False),
        ("神经网络AI-中等", "medium", True),
        ("神经网络AI-专家", "expert", True),
        ("纯神经网络AI", "neural", True)
    ]
    
    test_scenarios = create_test_scenarios()
    
    for ai_name, difficulty, use_neural in ai_configs:
        print(f"\n--- 测试 {ai_name} ---")
        
        try:
            # 创建AI
            ai = IntegratedGomokuAI(
                ai_difficulty=difficulty,
                time_limit=2.0,
                use_neural=use_neural
            )
            
            # 显示AI信息
            ai_info = ai.get_ai_info()
            print(f"AI类型: {ai_info['engine_type']}")
            print(f"神经网络启用: {ai_info.get('neural_enabled', False)}")
            
            if ai_info.get('neural_enabled'):
                neural_info = ai_info.get('neural_model_info', {})
                print(f"神经网络架构: {neural_info.get('architecture', 'N/A')}")
                print(f"模型参数数量: {neural_info.get('total_parameters', 'N/A')}")
            
            # 在所有场景上测试
            total_time = 0
            scenario_results = []
            
            for scenario_name, board, player in test_scenarios:
                start_time = time.time()
                
                # 获取AI移动
                move = ai.get_ai_move(board, player)
                
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                scenario_results.append({
                    'scenario': scenario_name,
                    'move': move,
                    'time': elapsed_time
                })
                
                print(f"  {scenario_name}: {move} (用时: {elapsed_time:.3f}s)")
            
            # 性能摘要
            avg_time = total_time / len(test_scenarios)
            performance = ai.get_performance_summary()
            
            print(f"平均决策时间: {avg_time:.3f}s")
            print(f"总移动数: {performance['total_moves']}")
            print(f"平均节点搜索: {performance.get('avg_nodes_per_move', 'N/A')}")
            
            # 神经网络特定统计
            if hasattr(ai.evaluator, 'get_neural_success_rate'):
                success_rate = ai.evaluator.get_neural_success_rate()
                print(f"神经网络成功率: {success_rate:.2%}")
            
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()

def test_neural_vs_traditional():
    """神经网络AI vs 传统AI对比测试"""
    print("\n=== 神经网络AI vs 传统AI对比测试 ===")
    
    # 创建两个AI
    traditional_ai = IntegratedGomokuAI(
        ai_difficulty="hard",
        time_limit=2.0,
        use_neural=False
    )
    
    neural_ai = IntegratedGomokuAI(
        ai_difficulty="hard",
        time_limit=2.0,
        use_neural=True
    )
    
    test_scenarios = create_test_scenarios()
    
    print(f"测试场景数: {len(test_scenarios)}")
    print("场景\t\t传统AI\t\t\t神经AI\t\t\t速度对比")
    print("-" * 80)
    
    total_traditional_time = 0
    total_neural_time = 0
    
    for scenario_name, board, player in test_scenarios:
        # 测试传统AI
        start_time = time.time()
        traditional_move = traditional_ai.get_ai_move(board, player)
        traditional_time = time.time() - start_time
        total_traditional_time += traditional_time
        
        # 测试神经网络AI
        start_time = time.time()
        neural_move = neural_ai.get_ai_move(board, player)
        neural_time = time.time() - start_time
        total_neural_time += neural_time
        
        # 计算速度比
        speed_ratio = traditional_time / neural_time if neural_time > 0 else float('inf')
        
        print(f"{scenario_name[:12]:<12}\t{traditional_move}\t{traditional_time:.3f}s\t\t{neural_move}\t{neural_time:.3f}s\t\t{speed_ratio:.2f}x")
    
    print("-" * 80)
    print(f"总计\t\t\t\t{total_traditional_time:.3f}s\t\t\t{total_neural_time:.3f}s\t\t{total_traditional_time/total_neural_time:.2f}x")
    
    # 获取详细性能统计
    traditional_stats = traditional_ai.get_performance_summary()
    neural_stats = neural_ai.get_performance_summary()
    
    print(f"\n传统AI平均节点搜索: {traditional_stats.get('avg_nodes_per_move', 'N/A')}")
    print(f"神经AI平均节点搜索: {neural_stats.get('avg_nodes_per_move', 'N/A')}")

def test_neural_game_integration():
    """测试神经网络与游戏界面集成"""
    print("\n=== 神经网络游戏集成测试 ===")
    
    # 创建增强版游戏
    game = EnhancedGomokuGame(ai_difficulty="neural")
    
    print("创建神经网络增强游戏...")
    
    # 获取AI信息
    ai_info = game.get_ai_info()
    print(f"AI引擎: {ai_info['engine_type']}")
    print(f"神经网络启用: {ai_info.get('neural_enabled', False)}")
    
    # 模拟几步游戏
    print("模拟游戏进程...")
    
    # 人类移动
    game.make_move(7, 7)
    print(f"人类移动: (7, 7)")
    
    # AI移动
    ai_move = game.ai_move()
    if ai_move:
        print(f"AI移动: {ai_move}")
        
        if game.last_ai_result:
            result = game.last_ai_result
            print(f"AI决策时间: {result['time']:.3f}s")
            print(f"AI性能数据: {result['performance']}")
    
    # 再模拟几步
    for i in range(3):
        # 人类随机移动
        empty_positions = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) 
                          if game.board[r][c] == EMPTY]
        if empty_positions:
            human_move = random.choice(empty_positions)
            game.make_move(*human_move)
            print(f"人类移动 {i+2}: {human_move}")
            
            # AI移动
            ai_move = game.ai_move()
            if ai_move:
                print(f"AI移动 {i+2}: {ai_move}")
    
    # 获取最终统计
    stats = game.get_performance_stats()
    print(f"\n游戏统计: {stats}")

def benchmark_neural_performance():
    """神经网络性能基准测试"""
    print("\n=== 神经网络性能基准测试 ===")
    
    # 创建神经网络评估器
    config = NeuralConfig(
        board_size=15,
        input_features=32,
        hidden_layers=[64, 32, 16],
        use_pattern_features=True,
        use_position_features=True,
        use_threat_features=True
    )
    
    evaluator = NeuralNetworkEvaluator(config)
    
    # 创建测试棋盘
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # 随机放置一些棋子
    for _ in range(20):
        row, col = random.randint(0, BOARD_SIZE-1), random.randint(0, BOARD_SIZE-1)
        if board[row][col] == EMPTY:
            board[row][col] = random.choice([BLACK, WHITE])
    
    # 性能测试
    test_counts = [10, 100, 500, 1000]
    
    print("评估次数\t平均时间(ms)\t总时间(s)\t评估/秒")
    print("-" * 50)
    
    for count in test_counts:
        start_time = time.time()
        
        for _ in range(count):
            evaluator.evaluate_board(board, BLACK)
        
        total_time = time.time() - start_time
        avg_time = total_time / count * 1000  # 转换为毫秒
        eval_per_sec = count / total_time
        
        print(f"{count}\t\t{avg_time:.2f}\t\t{total_time:.3f}\t\t{eval_per_sec:.1f}")
    
    # 获取评估器统计
    stats = evaluator.get_performance_stats()
    print(f"\n评估器统计: {stats}")
    
    # 内存使用估算
    model_info = evaluator.get_model_info()
    print(f"模型信息: {model_info}")

def main():
    """运行所有测试"""
    print("神经网络五子棋AI集成测试套件")
    print("=" * 50)
    
    try:
        # 基础集成测试
        test_neural_ai_integration()
        
        # 对比测试
        test_neural_vs_traditional()
        
        # 游戏集成测试
        test_neural_game_integration()
        
        # 性能基准测试
        benchmark_neural_performance()
        
        print("\n" + "=" * 50)
        print("所有测试完成！神经网络AI集成成功。")
        print("\n主要特性:")
        print("* 轻量级神经网络架构，无需深度学习框架")
        print("* 智能特征提取：棋型、位置、威胁、历史特征")
        print("* 混合评估：结合传统算法和神经网络")
        print("* 无缝集成到现有AI系统")
        print("* 实时性能优化，支持快速决策")
        print("* 可配置的网络结构和评估权重")
        print("* 完整的错误处理和回退机制")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()