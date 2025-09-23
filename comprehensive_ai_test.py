"""
AI系统综合测试和性能评估
完整测试新实现的深度学习AI系统

作者：Claude AI Engineer
日期：2025-09-23
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 导入所有AI模块
from integrated_ai import IntegratedGomokuAI, EnhancedGomokuGame, BOARD_SIZE, EMPTY, BLACK, WHITE
from neural_evaluator import NeuralNetworkEvaluator, FeatureExtractor
from neural_mcts import NeuralMCTS, AlphaZeroStyleNetwork, NeuralMCTSAdapter
from training_data_collector import TrainingDataCollector, SelfPlayEngine, TrainingDataManager
from modern_ai import ModernGomokuAI
from advanced_evaluator import ComprehensiveEvaluator

class AISystemTester:
    """AI系统综合测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_data = {}
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合测试套件"""
        print("=== AI系统综合测试开始 ===")
        
        # 1. 组件单元测试
        print("\n1. 组件单元测试")
        component_results = self._test_individual_components()
        
        # 2. 集成测试
        print("\n2. 系统集成测试")
        integration_results = self._test_system_integration()
        
        # 3. 性能对比测试
        print("\n3. AI性能对比测试")
        performance_results = self._test_ai_performance()
        
        # 4. 训练数据质量测试
        print("\n4. 训练数据质量测试")
        data_quality_results = self._test_training_data_quality()
        
        # 5. 稳定性测试
        print("\n5. 系统稳定性测试")
        stability_results = self._test_system_stability()
        
        # 汇总结果
        self.test_results = {
            'component_tests': component_results,
            'integration_tests': integration_results,
            'performance_tests': performance_results,
            'data_quality_tests': data_quality_results,
            'stability_tests': stability_results,
            'overall_score': self._calculate_overall_score(),
            'timestamp': time.time()
        }
        
        # 生成报告
        self._generate_test_report()
        
        print("\n=== AI系统综合测试完成 ===")
        return self.test_results
    
    def _test_individual_components(self) -> Dict[str, Any]:
        """测试各个组件"""
        results = {}
        
        # 测试神经网络评估器
        print("  测试神经网络评估器...")
        neural_eval_result = self._test_neural_evaluator()
        results['neural_evaluator'] = neural_eval_result
        
        # 测试Neural MCTS
        print("  测试Neural MCTS...")
        neural_mcts_result = self._test_neural_mcts()
        results['neural_mcts'] = neural_mcts_result
        
        # 测试训练数据收集器
        print("  测试训练数据收集器...")
        data_collector_result = self._test_data_collector()
        results['data_collector'] = data_collector_result
        
        return results
    
    def _test_neural_evaluator(self) -> Dict[str, Any]:
        """测试神经网络评估器"""
        try:
            # 创建评估器
            evaluator = NeuralNetworkEvaluator()
            
            # 创建测试棋盘
            board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            board[7][7] = BLACK
            board[7][8] = WHITE
            
            # 测试评估
            start_time = time.time()
            score = evaluator.evaluate_position(board, BLACK)
            eval_time = time.time() - start_time
            
            # 特征提取测试
            features = evaluator.feature_extractor.extract_features(board, BLACK)
            
            return {
                'status': 'passed',
                'evaluation_score': score,
                'evaluation_time': eval_time,
                'feature_count': len(features),
                'feature_range': [min(features), max(features)],
                'network_parameters': evaluator.get_model_info()['total_parameters']
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_neural_mcts(self) -> Dict[str, Any]:
        """测试Neural MCTS"""
        try:
            # 创建Neural MCTS
            network = AlphaZeroStyleNetwork()
            mcts = NeuralMCTS(network=network, num_simulations=100)
            
            # 创建测试棋盘
            board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            board[7][7] = BLACK
            
            # 测试搜索
            start_time = time.time()
            move = mcts.search(board, WHITE)
            search_time = time.time() - start_time
            
            # 获取搜索统计
            stats = mcts.get_search_stats()
            
            return {
                'status': 'passed',
                'best_move': move,
                'search_time': search_time,
                'nodes_explored': stats.get('nodes_explored', 0),
                'simulations': stats.get('simulations', 0),
                'tree_size': len(mcts.root.children) if mcts.root else 0
            }
            
        except Exception as e:
            return {
                'status': 'failed', 
                'error': str(e)
            }
    
    def _test_data_collector(self) -> Dict[str, Any]:
        """测试训练数据收集器"""
        try:
            # 创建数据收集器
            collector = TrainingDataCollector("test_data")
            
            # 模拟游戏记录
            game_id = collector.start_game_recording()
            
            # 记录几个位置
            board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            for i in range(3):
                board[7+i][7] = BLACK if i % 2 == 0 else WHITE
                move_probs = [0.1] * (BOARD_SIZE * BOARD_SIZE)
                move_probs[7*BOARD_SIZE + 7] = 0.9
                
                collector.record_position(
                    board_state=board,
                    current_player=BLACK,
                    move_probabilities=move_probs,
                    actual_move=(7, 7),
                    position_value=0.5,
                    search_time=0.1,
                    nodes_searched=100
                )
            
            # 完成记录
            collector.finish_game_recording(BLACK)
            
            # 获取统计
            stats = collector.get_stats_summary()
            
            return {
                'status': 'passed',
                'game_id': game_id,
                'positions_recorded': len(collector.position_buffer),
                'stats': stats['collection_stats'],
                'buffer_size_mb': stats['buffer_status']['buffer_size_mb']
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_system_integration(self) -> Dict[str, Any]:
        """测试系统集成"""
        results = {}
        
        print("  测试AI引擎集成...")
        
        # 测试所有AI难度级别
        difficulties = ['easy', 'medium', 'hard', 'expert', 'neural', 'neural_mcts']
        
        for difficulty in difficulties:
            try:
                # 创建集成AI系统
                ai_system = IntegratedGomokuAI(ai_difficulty=difficulty, time_limit=1.0)
                
                # 创建测试棋盘
                board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
                board[7][7] = BLACK
                
                # 测试AI移动
                start_time = time.time()
                move = ai_system.get_ai_move(board, WHITE)
                move_time = time.time() - start_time
                
                # 获取AI信息
                ai_info = ai_system.get_ai_info()
                performance = ai_system.get_performance_summary()
                
                results[difficulty] = {
                    'status': 'passed',
                    'move': move,
                    'move_time': move_time,
                    'ai_features': len(ai_info.get('features', [])),
                    'engine_type': ai_info.get('engine_type', 'unknown'),
                    'performance': performance
                }
                
            except Exception as e:
                results[difficulty] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _test_ai_performance(self) -> Dict[str, Any]:
        """测试AI性能对比"""
        print("  运行AI对战测试...")
        
        # 测试配置
        test_configs = [
            {'name': 'Traditional_Medium', 'difficulty': 'medium'},
            {'name': 'Traditional_Hard', 'difficulty': 'hard'},
            {'name': 'Neural_Enhanced', 'difficulty': 'neural'},
            {'name': 'Neural_MCTS', 'difficulty': 'neural_mcts'}
        ]
        
        results = {}
        
        for config in test_configs:
            try:
                ai_system = IntegratedGomokuAI(ai_difficulty=config['difficulty'], time_limit=2.0)
                
                # 运行性能测试
                perf_result = self._run_performance_benchmark(ai_system, config['name'])
                results[config['name']] = perf_result
                
            except Exception as e:
                results[config['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _run_performance_benchmark(self, ai_system: IntegratedGomokuAI, name: str) -> Dict[str, Any]:
        """运行性能基准测试"""
        test_positions = self._generate_test_positions(10)
        
        move_times = []
        move_qualities = []
        search_nodes = []
        
        for board, player in test_positions:
            start_time = time.time()
            move = ai_system.get_ai_move(board, player)
            move_time = time.time() - start_time
            
            if move:
                move_times.append(move_time)
                
                # 简单的移动质量评估（基于位置价值）
                try:
                    evaluator = ComprehensiveEvaluator()
                    board_copy = [row[:] for row in board]
                    board_copy[move[0]][move[1]] = player
                    quality = evaluator.evaluate_position(board_copy, player)
                    move_qualities.append(quality)
                except:
                    move_qualities.append(0)
                
                # 获取搜索节点数
                performance = ai_system.get_performance_summary()
                avg_nodes = performance.get('avg_nodes_per_move', 0)
                search_nodes.append(avg_nodes)
        
        return {
            'status': 'passed',
            'avg_move_time': np.mean(move_times),
            'move_time_std': np.std(move_times),
            'avg_move_quality': np.mean(move_qualities),
            'avg_search_nodes': np.mean(search_nodes),
            'positions_tested': len(test_positions),
            'successful_moves': len(move_times)
        }
    
    def _generate_test_positions(self, count: int) -> List[Tuple[List[List[int]], int]]:
        """生成测试位置"""
        positions = []
        
        for i in range(count):
            board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            
            # 随机放置一些棋子
            num_pieces = np.random.randint(5, 15)
            for _ in range(num_pieces):
                row = np.random.randint(0, BOARD_SIZE)
                col = np.random.randint(0, BOARD_SIZE)
                if board[row][col] == EMPTY:
                    board[row][col] = BLACK if np.random.random() < 0.5 else WHITE
            
            player = BLACK if i % 2 == 0 else WHITE
            positions.append((board, player))
        
        return positions
    
    def _test_training_data_quality(self) -> Dict[str, Any]:
        """测试训练数据质量"""
        try:
            print("  生成测试数据...")
            
            # 创建自我对弈引擎
            collector = TrainingDataCollector("test_training_data")
            engine = SelfPlayEngine(
                ai_config={'ai_difficulty': 'medium'},
                data_collector=collector
            )
            
            # 运行少量自我对弈
            records = engine.run_self_play_batch(num_games=3, verbose=False)
            
            # 分析数据质量
            manager = TrainingDataManager(collector)
            manager.analyze_data_distribution()
            
            stats = collector.get_stats_summary()
            
            return {
                'status': 'passed',
                'games_generated': len(records),
                'total_positions': len(collector.position_buffer),
                'data_quality': stats['data_quality'],
                'avg_game_length': stats['collection_stats']['avg_game_length'],
                'win_distribution': stats['collection_stats']['win_stats']
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_system_stability(self) -> Dict[str, Any]:
        """测试系统稳定性"""
        print("  运行稳定性测试...")
        
        stability_results = {
            'memory_leaks': self._test_memory_usage(),
            'error_handling': self._test_error_handling(),
            'concurrent_access': self._test_concurrent_access()
        }
        
        return stability_results
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # 运行多次AI操作
            ai_system = IntegratedGomokuAI(ai_difficulty='neural')
            board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            
            for i in range(20):
                board[7][7+i%5] = BLACK if i % 2 == 0 else WHITE
                ai_system.get_ai_move(board, WHITE)
                board[7][7+i%5] = EMPTY  # 清理
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            return {
                'status': 'passed',
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'memory_leak_detected': memory_increase > 50  # 50MB阈值
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        error_tests = []
        
        # 测试无效输入
        try:
            ai_system = IntegratedGomokuAI(ai_difficulty='neural')
            invalid_board = [[1, 2, 3]]  # 无效棋盘
            move = ai_system.get_ai_move(invalid_board, BLACK)
            error_tests.append({'test': 'invalid_board', 'status': 'handled'})
        except:
            error_tests.append({'test': 'invalid_board', 'status': 'failed'})
        
        # 测试无效AI配置
        try:
            ai_system = IntegratedGomokuAI(ai_difficulty='invalid_difficulty')
            error_tests.append({'test': 'invalid_config', 'status': 'handled'})
        except:
            error_tests.append({'test': 'invalid_config', 'status': 'failed'})
        
        passed_tests = sum(1 for test in error_tests if test['status'] == 'handled')
        
        return {
            'status': 'passed' if passed_tests == len(error_tests) else 'partial',
            'tests_passed': passed_tests,
            'total_tests': len(error_tests),
            'details': error_tests
        }
    
    def _test_concurrent_access(self) -> Dict[str, Any]:
        """测试并发访问"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker_thread(thread_id):
            try:
                ai_system = IntegratedGomokuAI(ai_difficulty='medium')
                board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
                board[7][7] = BLACK
                
                move = ai_system.get_ai_move(board, WHITE)
                results_queue.put({'thread_id': thread_id, 'status': 'success', 'move': move})
            except Exception as e:
                results_queue.put({'thread_id': thread_id, 'status': 'error', 'error': str(e)})
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        successful_threads = sum(1 for r in results if r['status'] == 'success')
        
        return {
            'status': 'passed' if successful_threads == 5 else 'partial',
            'successful_threads': successful_threads,
            'total_threads': 5,
            'details': results
        }
    
    def _calculate_overall_score(self) -> float:
        """计算总体分数"""
        # 简化的评分系统
        score = 0.0
        
        # 组件测试权重
        component_weight = 0.3
        integration_weight = 0.3
        performance_weight = 0.2
        stability_weight = 0.2
        
        # 计算各部分分数
        if 'component_tests' in self.test_results:
            component_score = self._calculate_component_score(self.test_results['component_tests'])
            score += component_score * component_weight
        
        if 'integration_tests' in self.test_results:
            integration_score = self._calculate_integration_score(self.test_results['integration_tests'])
            score += integration_score * integration_weight
        
        if 'performance_tests' in self.test_results:
            performance_score = self._calculate_performance_score(self.test_results['performance_tests'])
            score += performance_score * performance_weight
        
        if 'stability_tests' in self.test_results:
            stability_score = self._calculate_stability_score(self.test_results['stability_tests'])
            score += stability_score * stability_weight
        
        return min(score, 100.0)
    
    def _calculate_component_score(self, component_tests: Dict) -> float:
        """计算组件测试分数"""
        passed = sum(1 for test in component_tests.values() if test.get('status') == 'passed')
        total = len(component_tests)
        return (passed / total) * 100 if total > 0 else 0
    
    def _calculate_integration_score(self, integration_tests: Dict) -> float:
        """计算集成测试分数"""
        passed = sum(1 for test in integration_tests.values() if test.get('status') == 'passed')
        total = len(integration_tests)
        return (passed / total) * 100 if total > 0 else 0
    
    def _calculate_performance_score(self, performance_tests: Dict) -> float:
        """计算性能测试分数"""
        total_score = 0
        count = 0
        
        for test_name, test_result in performance_tests.items():
            if test_result.get('status') == 'passed':
                # 基于移动时间和质量评分
                move_time = test_result.get('avg_move_time', 5.0)
                move_quality = test_result.get('avg_move_quality', 0)
                
                # 时间分数 (越快越好，但有下限)
                time_score = max(0, 100 - move_time * 20)
                
                # 质量分数
                quality_score = max(0, min(100, move_quality * 100 + 50))
                
                test_score = (time_score + quality_score) / 2
                total_score += test_score
                count += 1
        
        return total_score / count if count > 0 else 0
    
    def _calculate_stability_score(self, stability_tests: Dict) -> float:
        """计算稳定性测试分数"""
        scores = []
        
        for test_name, test_result in stability_tests.items():
            if test_name == 'memory_leaks':
                if not test_result.get('memory_leak_detected', True):
                    scores.append(100)
                else:
                    scores.append(50)
            elif test_name in ['error_handling', 'concurrent_access']:
                if test_result.get('status') == 'passed':
                    scores.append(100)
                elif test_result.get('status') == 'partial':
                    scores.append(70)
                else:
                    scores.append(0)
        
        return np.mean(scores) if scores else 0
    
    def _generate_test_report(self):
        """生成测试报告"""
        report_path = Path("test_reports")
        report_path.mkdir(exist_ok=True)
        
        # 生成JSON报告
        json_report = report_path / f"ai_test_report_{int(time.time())}.json"
        with open(json_report, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成文本报告
        text_report = report_path / f"ai_test_summary_{int(time.time())}.txt"
        with open(text_report, 'w', encoding='utf-8') as f:
            self._write_text_report(f)
        
        print(f"\n测试报告已生成:")
        print(f"  JSON报告: {json_report}")
        print(f"  文本报告: {text_report}")
        
        # 尝试生成可视化报告
        try:
            self._generate_visual_report(report_path)
        except ImportError:
            print("  注意: 无法生成可视化报告 (需要matplotlib)")
    
    def _write_text_report(self, file):
        """写入文本报告"""
        file.write("=== AI系统综合测试报告 ===\n")
        file.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.test_results['timestamp']))}\n")
        file.write(f"总体分数: {self.test_results['overall_score']:.1f}/100\n\n")
        
        # 组件测试结果
        file.write("## 组件测试结果\n")
        for component, result in self.test_results['component_tests'].items():
            status = result['status']
            file.write(f"  {component}: {status.upper()}\n")
            if status == 'passed' and component == 'neural_evaluator':
                file.write(f"    - 评估时间: {result.get('evaluation_time', 0):.4f}秒\n")
                file.write(f"    - 特征数量: {result.get('feature_count', 0)}\n")
                file.write(f"    - 网络参数: {result.get('network_parameters', 0)}\n")
        
        # 集成测试结果
        file.write("\n## 集成测试结果\n")
        for difficulty, result in self.test_results['integration_tests'].items():
            status = result['status']
            file.write(f"  {difficulty}: {status.upper()}\n")
            if status == 'passed':
                file.write(f"    - 移动时间: {result.get('move_time', 0):.3f}秒\n")
                file.write(f"    - 引擎类型: {result.get('engine_type', 'unknown')}\n")
        
        # 性能测试结果
        file.write("\n## 性能测试结果\n")
        for ai_name, result in self.test_results['performance_tests'].items():
            status = result['status']
            file.write(f"  {ai_name}: {status.upper()}\n")
            if status == 'passed':
                file.write(f"    - 平均移动时间: {result.get('avg_move_time', 0):.3f}秒\n")
                file.write(f"    - 平均移动质量: {result.get('avg_move_quality', 0):.3f}\n")
                file.write(f"    - 平均搜索节点: {result.get('avg_search_nodes', 0):.0f}\n")
        
        # 稳定性测试结果
        file.write("\n## 稳定性测试结果\n")
        stability = self.test_results['stability_tests']
        
        memory_test = stability.get('memory_leaks', {})
        if memory_test.get('status') == 'passed':
            file.write(f"  内存测试: {'PASSED' if not memory_test.get('memory_leak_detected', True) else 'WARNING'}\n")
            file.write(f"    - 内存增长: {memory_test.get('memory_increase_mb', 0):.1f}MB\n")
        
        error_test = stability.get('error_handling', {})
        if error_test.get('status') in ['passed', 'partial']:
            file.write(f"  错误处理: {error_test['status'].upper()}\n")
            file.write(f"    - 通过测试: {error_test.get('tests_passed', 0)}/{error_test.get('total_tests', 0)}\n")
        
        concurrent_test = stability.get('concurrent_access', {})
        if concurrent_test.get('status') in ['passed', 'partial']:
            file.write(f"  并发访问: {concurrent_test['status'].upper()}\n")
            file.write(f"    - 成功线程: {concurrent_test.get('successful_threads', 0)}/{concurrent_test.get('total_threads', 0)}\n")
    
    def _generate_visual_report(self, report_path: Path):
        """生成可视化报告"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建性能对比图
            if 'performance_tests' in self.test_results:
                self._plot_performance_comparison(report_path)
            
            # 创建组件状态图
            self._plot_component_status(report_path)
            
            print(f"  可视化报告: {report_path / 'performance_comparison.png'}")
            print(f"  组件状态图: {report_path / 'component_status.png'}")
            
        except Exception as e:
            print(f"  可视化报告生成失败: {e}")
    
    def _plot_performance_comparison(self, report_path: Path):
        """绘制性能对比图"""
        performance_data = self.test_results['performance_tests']
        
        names = []
        move_times = []
        move_qualities = []
        
        for name, data in performance_data.items():
            if data.get('status') == 'passed':
                names.append(name.replace('_', ' '))
                move_times.append(data.get('avg_move_time', 0))
                move_qualities.append(data.get('avg_move_quality', 0))
        
        if names:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 移动时间对比
            ax1.bar(names, move_times, color='skyblue')
            ax1.set_title('平均移动时间对比')
            ax1.set_ylabel('时间 (秒)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 移动质量对比
            ax2.bar(names, move_qualities, color='lightgreen')
            ax2.set_title('平均移动质量对比')
            ax2.set_ylabel('质量分数')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(report_path / 'performance_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _plot_component_status(self, report_path: Path):
        """绘制组件状态图"""
        # 统计各测试类型的通过率
        test_categories = ['组件测试', '集成测试', '性能测试', '稳定性测试']
        pass_rates = []
        
        # 组件测试通过率
        component_tests = self.test_results.get('component_tests', {})
        component_passed = sum(1 for t in component_tests.values() if t.get('status') == 'passed')
        component_total = len(component_tests)
        pass_rates.append(component_passed / component_total * 100 if component_total > 0 else 0)
        
        # 集成测试通过率
        integration_tests = self.test_results.get('integration_tests', {})
        integration_passed = sum(1 for t in integration_tests.values() if t.get('status') == 'passed')
        integration_total = len(integration_tests)
        pass_rates.append(integration_passed / integration_total * 100 if integration_total > 0 else 0)
        
        # 性能测试通过率
        performance_tests = self.test_results.get('performance_tests', {})
        performance_passed = sum(1 for t in performance_tests.values() if t.get('status') == 'passed')
        performance_total = len(performance_tests)
        pass_rates.append(performance_passed / performance_total * 100 if performance_total > 0 else 0)
        
        # 稳定性测试通过率
        stability_tests = self.test_results.get('stability_tests', {})
        stability_passed = sum(1 for t in stability_tests.values() 
                             if t.get('status') in ['passed', 'partial'])
        stability_total = len(stability_tests)
        pass_rates.append(stability_passed / stability_total * 100 if stability_total > 0 else 0)
        
        # 绘制图表
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(test_categories, pass_rates, color=colors)
        
        ax.set_title('AI系统测试结果概览', fontsize=16, pad=20)
        ax.set_ylabel('通过率 (%)', fontsize=12)
        ax.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, rate in zip(bars, pass_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
        
        # 添加总体分数
        overall_score = self.test_results.get('overall_score', 0)
        ax.text(0.02, 0.98, f'总体分数: {overall_score:.1f}/100', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(report_path / 'component_status.png', dpi=150, bbox_inches='tight')
        plt.close()

def run_quick_test():
    """运行快速测试"""
    print("=== 快速AI系统测试 ===")
    
    tester = AISystemTester()
    
    # 只运行基础测试
    print("测试神经网络评估器...")
    neural_result = tester._test_neural_evaluator()
    print(f"结果: {neural_result['status']}")
    
    print("测试系统集成...")
    # 测试几个主要难度
    for difficulty in ['medium', 'neural', 'neural_mcts']:
        try:
            ai_system = IntegratedGomokuAI(ai_difficulty=difficulty)
            board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            board[7][7] = BLACK
            
            start_time = time.time()
            move = ai_system.get_ai_move(board, WHITE)
            move_time = time.time() - start_time
            
            print(f"  {difficulty}: OK (移动: {move}, 时间: {move_time:.3f}s)")
            
        except Exception as e:
            print(f"  {difficulty}: 错误 - {e}")
    
    print("快速测试完成!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        run_quick_test()
    else:
        # 运行完整测试
        tester = AISystemTester()
        results = tester.run_comprehensive_test()
        
        print(f"\n=== 测试完成 ===")
        print(f"总体分数: {results['overall_score']:.1f}/100")
        
        # 显示主要结果
        passed_components = sum(1 for t in results['component_tests'].values() 
                               if t.get('status') == 'passed')
        total_components = len(results['component_tests'])
        
        passed_integrations = sum(1 for t in results['integration_tests'].values() 
                                 if t.get('status') == 'passed')
        total_integrations = len(results['integration_tests'])
        
        print(f"组件测试: {passed_components}/{total_components} 通过")
        print(f"集成测试: {passed_integrations}/{total_integrations} 通过")
        
        # 推荐后续步骤
        if results['overall_score'] >= 80:
            print("\n✅ AI系统运行良好，可以开始深度学习训练！")
        elif results['overall_score'] >= 60:
            print("\n⚠️  AI系统基本正常，建议优化后再进行训练")
        else:
            print("\n❌ AI系统存在问题，需要修复后再使用")