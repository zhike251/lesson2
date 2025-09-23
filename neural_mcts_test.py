# -*- coding: utf-8 -*-
"""
Neural MCTS性能测试脚本
对比传统Minimax、普通MCTS和Neural MCTS的性能

作者：Claude AI Engineer
日期：2025-09-22
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

class NeuralMCTSPerformanceTester:
    """Neural MCTS性能测试器"""
    
    def __init__(self):
        self.test_results = []
        self.test_positions = []
        
    def create_test_positions(self) -> List[List[List[int]]]:
        """创建测试棋局位置"""
        positions = []
        
        # 测试位置1：开局阶段
        board1 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board1[7][7] = BLACK
        board1[7][8] = WHITE
        board1[8][7] = BLACK
        positions.append(board1)
        
        # 测试位置2：中局阶段
        board2 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        # 创建一个复杂的中局位置
        moves = [
            (7, 7, BLACK), (7, 8, WHITE), (8, 7, BLACK), (6, 8, WHITE),
            (8, 6, BLACK), (9, 7, WHITE), (6, 7, BLACK), (7, 6, WHITE),
            (5, 7, BLACK), (8, 8, WHITE), (9, 6, BLACK), (6, 6, WHITE)
        ]
        for row, col, player in moves:
            board2[row][col] = player
        positions.append(board2)
        
        # 测试位置3：危急情况
        board3 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        # 创建一个有威胁的局面
        board3[7][7] = BLACK
        board3[7][8] = BLACK
        board3[7][9] = BLACK
        board3[7][10] = BLACK  # 黑棋四连，需要白棋阻挡
        board3[6][7] = WHITE
        board3[8][7] = WHITE
        board3[6][8] = WHITE
        positions.append(board3)
        
        return positions
    
    def test_ai_engine(self, engine_name: str, ai_system, positions: List[List[List[int]]], 
                      num_trials: int = 3) -> Dict:
        """
        测试AI引擎性能
        
        Args:
            engine_name: 引擎名称
            ai_system: AI系统对象
            positions: 测试位置列表
            num_trials: 每个位置的测试次数
            
        Returns:
            测试结果字典
        """
        print(f"\n测试引擎: {engine_name}")
        print("=" * 50)
        
        results = {
            'engine_name': engine_name,
            'positions': [],
            'total_time': 0,
            'total_moves': 0,
            'avg_time_per_move': 0,
            'quality_scores': []
        }
        
        for pos_idx, board in enumerate(positions):
            print(f"\n位置 {pos_idx + 1}: ", end="")
            position_results = {
                'position_id': pos_idx + 1,
                'trials': [],
                'avg_time': 0,
                'move_consistency': 0
            }
            
            trial_times = []
            trial_moves = []
            
            for trial in range(num_trials):
                print(f"试验{trial + 1} ", end="")
                
                start_time = time.time()
                try:
                    # 获取AI移动
                    if hasattr(ai_system, 'get_ai_move'):\n                        move = ai_system.get_ai_move(board, WHITE)\n                    else:\n                        result = ai_system.get_best_move(board, WHITE)\n                        move = result.move if hasattr(result, 'move') else result\n                    \n                    elapsed_time = time.time() - start_time\n                    \n                    trial_times.append(elapsed_time)\n                    trial_moves.append(move)\n                    \n                    position_results['trials'].append({\n                        'trial': trial + 1,\n                        'move': move,\n                        'time': elapsed_time\n                    })\n                    \n                    print(f\"({move}, {elapsed_time:.2f}s) \", end=\"\")\n                    \n                except Exception as e:\n                    print(f\"错误: {e} \", end=\"\")\n                    trial_times.append(float('inf'))\n                    trial_moves.append(None)\n            \n            # 计算位置平均时间\n            valid_times = [t for t in trial_times if t != float('inf')]\n            position_results['avg_time'] = sum(valid_times) / len(valid_times) if valid_times else float('inf')\n            \n            # 计算移动一致性（相同移动的比例）\n            valid_moves = [m for m in trial_moves if m is not None]\n            if valid_moves:\n                most_common_move = max(set(valid_moves), key=valid_moves.count)\n                consistency = valid_moves.count(most_common_move) / len(valid_moves)\n                position_results['move_consistency'] = consistency\n            \n            results['positions'].append(position_results)\n            results['total_time'] += position_results['avg_time']\n            results['total_moves'] += len(valid_moves)\n            \n            print(f\"\\n  平均时间: {position_results['avg_time']:.2f}s\")\n            print(f\"  移动一致性: {position_results['move_consistency']:.2%}\")\n        \n        # 计算总体统计\n        if results['total_moves'] > 0:\n            results['avg_time_per_move'] = results['total_time'] / len(positions)\n        \n        return results\n    \n    def compare_engines(self) -> Dict:\n        \"\"\"\n        比较不同AI引擎的性能\n        \n        Returns:\n            比较结果字典\n        \"\"\"\n        print(\"开始Neural MCTS性能比较测试...\")\n        print(\"\\n创建测试位置...\")\n        \n        positions = self.create_test_positions()\n        print(f\"创建了 {len(positions)} 个测试位置\")\n        \n        # 测试配置\n        test_configs = [\n            {\n                'name': 'Traditional Minimax (Medium)',\n                'difficulty': 'medium',\n                'engine_type': 'minimax'\n            },\n            {\n                'name': 'Traditional Minimax + Neural (Expert)',\n                'difficulty': 'expert',\n                'engine_type': 'minimax'\n            },\n            {\n                'name': 'Neural MCTS (Default)',\n                'difficulty': 'neural_mcts',\n                'engine_type': 'neural_mcts'\n            }\n        ]\n        \n        comparison_results = {\n            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),\n            'test_positions_count': len(positions),\n            'engines': []\n        }\n        \n        # 测试每个引擎\n        for config in test_configs:\n            try:\n                print(f\"\\n\\n{'='*60}\")\n                print(f\"正在创建引擎: {config['name']}\")\n                \n                # 创建AI系统\n                if config['engine_type'] == 'neural_mcts':\n                    # 创建Neural MCTS系统\n                    neural_config = NeuralConfig(\n                        board_size=15,\n                        input_features=32,\n                        hidden_layers=[64, 32],\n                        use_pattern_features=True,\n                        use_position_features=True\n                    )\n                    neural_evaluator = NeuralNetworkEvaluator(neural_config)\n                    network_adapter = NeuralEvaluatorNetworkAdapter(neural_evaluator)\n                    \n                    ai_system = NeuralMCTSAdapter(\n                        neural_network=network_adapter,\n                        mcts_simulations=400,  # 减少模拟次数以加快测试\n                        c_puct=1.25,\n                        time_limit=3.0\n                    )\n                else:\n                    # 创建传统AI系统\n                    ai_system = IntegratedGomokuAI(\n                        ai_difficulty=config['difficulty'],\n                        time_limit=3.0,\n                        engine_type=config['engine_type']\n                    )\n                \n                # 测试引擎\n                engine_results = self.test_ai_engine(\n                    config['name'], \n                    ai_system, \n                    positions, \n                    num_trials=2  # 减少试验次数以加快测试\n                )\n                \n                comparison_results['engines'].append(engine_results)\n                \n            except Exception as e:\n                print(f\"测试引擎 {config['name']} 时出错: {e}\")\n                error_result = {\n                    'engine_name': config['name'],\n                    'error': str(e),\n                    'total_time': float('inf'),\n                    'avg_time_per_move': float('inf')\n                }\n                comparison_results['engines'].append(error_result)\n        \n        return comparison_results\n    \n    def print_comparison_summary(self, results: Dict):\n        \"\"\"\n        打印比较结果摘要\n        \n        Args:\n            results: 比较结果字典\n        \"\"\"\n        print(\"\\n\\n\" + \"=\"*80)\n        print(\"Neural MCTS 性能比较总结\")\n        print(\"=\"*80)\n        \n        print(f\"\\n测试时间: {results['test_timestamp']}\")\n        print(f\"测试位置数量: {results['test_positions_count']}\")\n        \n        print(\"\\n引擎性能排名 (按平均响应时间):\")\n        print(\"-\" * 60)\n        \n        # 按平均时间排序\n        valid_engines = [e for e in results['engines'] if 'error' not in e]\n        sorted_engines = sorted(valid_engines, key=lambda x: x['avg_time_per_move'])\n        \n        for i, engine in enumerate(sorted_engines, 1):\n            print(f\"{i}. {engine['engine_name']}:\")\n            print(f\"   平均响应时间: {engine['avg_time_per_move']:.2f}s\")\n            print(f\"   总测试时间: {engine['total_time']:.2f}s\")\n            \n            # 显示每个位置的详细信息\n            if 'positions' in engine:\n                for pos in engine['positions']:\n                    print(f\"   位置 {pos['position_id']}: {pos['avg_time']:.2f}s, 一致性: {pos['move_consistency']:.1%}\")\n            print()\n        \n        # 显示错误的引擎\n        error_engines = [e for e in results['engines'] if 'error' in e]\n        if error_engines:\n            print(\"\\n测试失败的引擎:\")\n            print(\"-\" * 40)\n            for engine in error_engines:\n                print(f\"- {engine['engine_name']}: {engine['error']}\")\n    \n    def save_results(self, results: Dict, filename: str = \"neural_mcts_test_results.json\"):\n        \"\"\"\n        保存测试结果到文件\n        \n        Args:\n            results: 测试结果\n            filename: 保存文件名\n        \"\"\"\n        try:\n            with open(filename, 'w', encoding='utf-8') as f:\n                json.dump(results, f, indent=2, ensure_ascii=False, default=str)\n            print(f\"\\n测试结果已保存到: {filename}\")\n        except Exception as e:\n            print(f\"保存测试结果失败: {e}\")\n\ndef test_neural_mcts_basic():\n    \"\"\"\n    基本的Neural MCTS功能测试\n    \"\"\"\n    print(\"开始Neural MCTS基本功能测试...\")\n    \n    try:\n        # 创建Neural MCTS系统\n        print(\"1. 创建Neural MCTS系统...\")\n        network = AlphaZeroStyleNetwork()\n        mcts_adapter = NeuralMCTSAdapter(\n            neural_network=network,\n            mcts_simulations=100,\n            c_puct=1.0\n        )\n        print(\"   ✓ Neural MCTS系统创建成功\")\n        \n        # 测试基本移动\n        print(\"\\n2. 测试基本移动功能...\")\n        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]\n        board[7][7] = BLACK\n        \n        start_time = time.time()\n        result = mcts_adapter.get_best_move(board, WHITE)\n        elapsed_time = time.time() - start_time\n        \n        if hasattr(result, 'move') and result.move:\n            print(f\"   ✓ 获取移动成功: {result.move}, 用时: {elapsed_time:.2f}s\")\n            print(f\"   分数: {result.score}, 搜索节点: {result.nodes_searched}\")\n        else:\n            print(\"   ✗ 获取移动失败\")\n        \n        # 测试评估功能\n        print(\"\\n3. 测试棋盘评估功能...\")\n        score = mcts_adapter.evaluate_board(board, WHITE)\n        print(f\"   ✓ 棋盘评估分数: {score}\")\n        \n        # 测试统计信息\n        print(\"\\n4. 测试统计信息...\")\n        stats = mcts_adapter.get_statistics()\n        print(f\"   ✓ 统计信息: {stats}\")\n        \n        print(\"\\n✓ Neural MCTS基本功能测试通过！\")\n        return True\n        \n    except Exception as e:\n        print(f\"\\n✗ Neural MCTS基本功能测试失败: {e}\")\n        import traceback\n        traceback.print_exc()\n        return False\n\ndef main():\n    \"\"\"\n    主测试函数\n    \"\"\"\n    print(\"Neural MCTS 测试程序\")\n    print(\"=\" * 40)\n    \n    # 基本功能测试\n    print(\"\\n阶段1: 基本功能测试\")\n    if not test_neural_mcts_basic():\n        print(\"基本功能测试失败，跳过性能比较测试\")\n        return\n    \n    # 性能比较测试\n    print(\"\\n\\n阶段2: 性能比较测试\")\n    print(\"注意：这可能需要几分钟时间...\")\n    \n    tester = NeuralMCTSPerformanceTester()\n    \n    try:\n        results = tester.compare_engines()\n        tester.print_comparison_summary(results)\n        tester.save_results(results)\n        \n        print(\"\\n✓ 所有测试完成！\")\n        \n    except Exception as e:\n        print(f\"\\n✗ 性能比较测试失败: {e}\")\n        import traceback\n        traceback.print_exc()\n\nif __name__ == \"__main__\":\n    main()\n