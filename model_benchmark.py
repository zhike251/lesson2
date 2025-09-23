"""
五子棋AI模型性能基准测试工具

提供全面的模型性能评估，包括：
1. 棋力评估（ELO等级评分）
2. 推理性能测试
3. 内存使用分析
4. 对战胜率统计
5. 决策质量评估

作者：Claude AI Engineer
日期：2025-09-22
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import logging

from deep_learning_architecture import GomokuNetwork, ModelConfig

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 对战测试
    num_games: int = 100
    max_moves_per_game: int = 225
    time_limit_per_move: float = 5.0
    
    # 性能测试
    inference_iterations: int = 1000
    memory_test_batches: int = 10
    
    # 质量评估
    test_positions_file: str = "test_positions.json"
    expert_moves_file: str = "expert_moves.json"
    
    # 输出设置
    save_results: bool = True
    plot_results: bool = True
    results_dir: str = "benchmark_results"

class EloRatingSystem:
    """ELO等级评分系统"""
    
    def __init__(self, k_factor: int = 32, initial_rating: int = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
    
    def get_rating(self, player: str) -> int:
        """获取玩家等级分"""
        return self.ratings.get(player, self.initial_rating)
    
    def expected_score(self, rating_a: int, rating_b: int) -> float:
        """计算期望得分"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, player_a: str, player_b: str, score_a: float):
        """更新等级分"""
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - score_a) - expected_b)
        
        self.ratings[player_a] = round(new_rating_a)
        self.ratings[player_b] = round(new_rating_b)

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.inference_times = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.cpu_usage = []
    
    def start_profiling(self):
        """开始性能分析"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.start_gpu_memory = torch.cuda.memory_allocated()
    
    def end_profiling(self):
        """结束性能分析"""
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        inference_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_delta)
        
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_delta = end_gpu_memory - self.start_gpu_memory
            self.gpu_memory_usage.append(gpu_memory_delta)
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage.append(cpu_percent)
    
    def get_stats(self) -> Dict:
        """获取性能统计"""
        stats = {
            'inference_time': {
                'mean': np.mean(self.inference_times) if self.inference_times else 0,
                'std': np.std(self.inference_times) if self.inference_times else 0,
                'min': np.min(self.inference_times) if self.inference_times else 0,
                'max': np.max(self.inference_times) if self.inference_times else 0,
                'median': np.median(self.inference_times) if self.inference_times else 0
            },
            'memory_usage': {
                'mean': np.mean(self.memory_usage) if self.memory_usage else 0,
                'std': np.std(self.memory_usage) if self.memory_usage else 0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0
            },
            'cpu_usage': {
                'mean': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max': np.max(self.cpu_usage) if self.cpu_usage else 0
            }
        }
        
        if torch.cuda.is_available() and self.gpu_memory_usage:
            stats['gpu_memory_usage'] = {
                'mean': np.mean(self.gpu_memory_usage),
                'std': np.std(self.gpu_memory_usage), 
                'max': np.max(self.gpu_memory_usage)
            }
        
        return stats

class DecisionQualityEvaluator:
    """决策质量评估器"""
    
    def __init__(self):
        self.expert_moves = {}
        self.test_positions = {}
        self.load_test_data()
    
    def load_test_data(self):
        """加载测试数据"""
        try:
            # 加载专家棋谱
            expert_file = Path("expert_moves.json")
            if expert_file.exists():
                with open(expert_file, 'r', encoding='utf-8') as f:
                    self.expert_moves = json.load(f)
            
            # 加载测试局面
            positions_file = Path("test_positions.json")
            if positions_file.exists():
                with open(positions_file, 'r', encoding='utf-8') as f:
                    self.test_positions = json.load(f)
        except Exception as e:
            logging.warning(f"无法加载测试数据: {e}")
    
    def evaluate_move_quality(self, model: GomokuNetwork, 
                             state: np.ndarray, player: int,
                             expert_move: Tuple[int, int] = None) -> Dict:
        """评估移动质量"""
        model.eval()
        
        with torch.no_grad():
            policy_dict, value = model.predict(state, player)
        
        # 策略分析
        if policy_dict:
            top_moves = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            top_move = top_moves[0][0]
            top_prob = top_moves[0][1]
            
            # 计算策略集中度（熵）
            probs = list(policy_dict.values())
            entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
            
            # 与专家移动的一致性
            expert_consistency = 0.0
            if expert_move and expert_move in policy_dict:
                expert_prob = policy_dict[expert_move]
                expert_rank = sorted(policy_dict.values(), reverse=True).index(expert_prob) + 1
                expert_consistency = expert_prob
            else:
                expert_rank = len(policy_dict) + 1
        else:
            top_moves = []
            top_move = None
            top_prob = 0.0
            entropy = 0.0
            expert_consistency = 0.0
            expert_rank = 0
        
        return {
            'top_move': top_move,
            'top_probability': top_prob,
            'top_5_moves': top_moves,
            'policy_entropy': entropy,
            'value_estimate': value,
            'expert_consistency': expert_consistency,
            'expert_move_rank': expert_rank,
            'num_legal_moves': len(policy_dict)
        }
    
    def batch_evaluate_positions(self, model: GomokuNetwork) -> Dict:
        """批量评估测试局面"""
        if not self.test_positions:
            return {}
        
        results = []
        
        for position_id, position_data in self.test_positions.items():
            state = np.array(position_data['state'])
            player = position_data['player']
            expert_move = tuple(position_data.get('expert_move', []))
            
            quality = self.evaluate_move_quality(model, state, player, expert_move)
            quality['position_id'] = position_id
            results.append(quality)
        
        # 汇总统计
        summary = {
            'avg_expert_consistency': np.mean([r['expert_consistency'] for r in results]),
            'avg_entropy': np.mean([r['policy_entropy'] for r in results]),
            'avg_value_estimate': np.mean([abs(r['value_estimate']) for r in results]),
            'expert_top1_rate': sum(1 for r in results if r['expert_move_rank'] == 1) / len(results),
            'expert_top3_rate': sum(1 for r in results if r['expert_move_rank'] <= 3) / len(results),
            'expert_top5_rate': sum(1 for r in results if r['expert_move_rank'] <= 5) / len(results),
            'total_positions': len(results)
        }
        
        return {
            'summary': summary,
            'detailed_results': results
        }

class GameSimulator:
    """游戏模拟器"""
    
    def __init__(self):
        pass
    
    def play_game(self, model1: GomokuNetwork, model2: GomokuNetwork,
                  max_moves: int = 225, time_limit: float = 5.0) -> Dict:
        """模拟两个模型对弈"""
        state = np.zeros((15, 15), dtype=int)
        current_player = 1
        move_history = []
        models = {1: model1, 2: model2}
        
        game_log = {
            'moves': [],
            'winner': 0,
            'total_moves': 0,
            'game_length': 0,
            'timeout': False
        }
        
        start_time = time.time()
        
        for move_count in range(max_moves):
            move_start_time = time.time()
            
            # 获取当前模型
            current_model = models[current_player]
            
            # 获取移动
            try:
                current_model.eval()
                with torch.no_grad():
                    policy_dict, value = current_model.predict(state, current_player, move_history)
                
                if not policy_dict:
                    break  # 没有合法移动
                
                # 选择最佳移动
                best_move = max(policy_dict.items(), key=lambda x: x[1])[0]
                row, col = best_move
                
                # 检查移动时间
                move_time = time.time() - move_start_time
                if move_time > time_limit:
                    game_log['timeout'] = True
                    game_log['winner'] = 3 - current_player  # 对手获胜
                    break
                
                # 执行移动
                state[row, col] = current_player
                move_history.append((row, col))
                
                # 记录移动
                game_log['moves'].append({
                    'player': current_player,
                    'move': (row, col),
                    'value_estimate': value,
                    'move_time': move_time,
                    'top_probability': policy_dict[best_move]
                })
                
                # 检查胜利
                if self._check_winner(state, row, col):
                    game_log['winner'] = current_player
                    break
                
                # 切换玩家
                current_player = 3 - current_player
                
            except Exception as e:
                logging.error(f"游戏模拟错误: {e}")
                break
        
        game_log['total_moves'] = len(game_log['moves'])
        game_log['game_length'] = time.time() - start_time
        
        return game_log
    
    def _check_winner(self, state: np.ndarray, row: int, col: int) -> bool:
        """检查是否获胜"""
        player = state[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while (0 <= x < 15 and 0 <= y < 15 and state[x, y] == player):
                count += 1
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while (0 <= x < 15 and 0 <= y < 15 and state[x, y] == player):
                count += 1
                x -= dx
                y -= dy
            
            if count >= 5:
                return True
        
        return False

class ModelBenchmark:
    """模型基准测试主类"""
    
    def __init__(self, model: GomokuNetwork, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 组件初始化
        self.profiler = PerformanceProfiler()
        self.quality_evaluator = DecisionQualityEvaluator()
        self.game_simulator = GameSimulator()
        self.elo_system = EloRatingSystem()
        
        # 结果存储
        self.results = {}
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 创建结果目录
        if self.config.save_results:
            Path(self.config.results_dir).mkdir(exist_ok=True)
    
    def benchmark_inference_speed(self) -> Dict:
        """基准测试推理速度"""
        self.logger.info("开始推理速度测试...")
        
        self.model.eval()
        self.profiler.reset()
        
        # 准备测试数据
        test_states = []
        for _ in range(self.config.inference_iterations):
            state = np.random.randint(0, 3, (15, 15))
            test_states.append(state)
        
        # 预热
        for _ in range(10):
            state = test_states[0]
            with torch.no_grad():
                self.model.predict(state, 1)
        
        # 正式测试
        for state in test_states:
            self.profiler.start_profiling()
            
            with torch.no_grad():
                policy, value = self.model.predict(state, 1)
            
            self.profiler.end_profiling()
        
        stats = self.profiler.get_stats()
        
        # 计算每秒推理次数
        avg_time = stats['inference_time']['mean']
        inferences_per_second = 1.0 / avg_time if avg_time > 0 else 0
        
        result = {
            'performance_stats': stats,
            'inferences_per_second': inferences_per_second,
            'total_iterations': self.config.inference_iterations
        }
        
        self.logger.info(f"推理速度测试完成: {inferences_per_second:.2f} inferences/sec")
        return result
    
    def benchmark_memory_usage(self) -> Dict:
        """基准测试内存使用"""
        self.logger.info("开始内存使用测试...")
        
        self.model.eval()
        
        # 测试不同批次大小的内存使用
        batch_sizes = [1, 4, 8, 16, 32, 64]
        memory_results = {}
        
        for batch_size in batch_sizes:
            try:
                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 准备批次数据
                batch_states = torch.randn(batch_size, 12, 15, 15).to(self.device)
                
                # 记录开始内存
                start_memory = psutil.virtual_memory().used
                if torch.cuda.is_available():
                    start_gpu_memory = torch.cuda.memory_allocated()
                
                # 推理
                with torch.no_grad():
                    policy, value = self.model(batch_states)
                
                # 记录结束内存
                end_memory = psutil.virtual_memory().used
                memory_delta = end_memory - start_memory
                
                memory_per_sample = memory_delta / batch_size
                
                result = {
                    'total_memory_delta': memory_delta,
                    'memory_per_sample': memory_per_sample
                }
                
                if torch.cuda.is_available():
                    end_gpu_memory = torch.cuda.memory_allocated()
                    gpu_memory_delta = end_gpu_memory - start_gpu_memory
                    result['gpu_memory_delta'] = gpu_memory_delta
                    result['gpu_memory_per_sample'] = gpu_memory_delta / batch_size
                
                memory_results[batch_size] = result
                
            except RuntimeError as e:
                self.logger.warning(f"批次大小 {batch_size} 内存不足: {e}")
                memory_results[batch_size] = {'error': str(e)}
        
        self.logger.info("内存使用测试完成")
        return memory_results
    
    def benchmark_decision_quality(self) -> Dict:
        """基准测试决策质量"""
        self.logger.info("开始决策质量评估...")
        
        # 批量评估测试局面
        quality_results = self.quality_evaluator.batch_evaluate_positions(self.model)
        
        if quality_results:
            self.logger.info(f"决策质量评估完成: "
                           f"专家一致性 {quality_results['summary']['avg_expert_consistency']:.3f}, "
                           f"Top-1命中率 {quality_results['summary']['expert_top1_rate']:.3f}")
        else:
            self.logger.warning("没有可用的测试局面数据")
        
        return quality_results
    
    def benchmark_vs_baselines(self, baseline_models: Dict[str, GomokuNetwork] = None) -> Dict:
        """基准测试对战基线模型"""
        self.logger.info("开始对战基线模型测试...")
        
        if not baseline_models:
            # 创建简单的基线模型
            baseline_models = {
                'random': self._create_random_model(),
                'simple': self._create_simple_model()
            }
        
        battle_results = {}
        
        for baseline_name, baseline_model in baseline_models.items():
            self.logger.info(f"对战 {baseline_name} 模型...")
            
            wins = losses = draws = 0
            games_data = []
            
            for game_idx in range(self.config.num_games):
                # 随机选择先后手
                if game_idx % 2 == 0:
                    game_result = self.game_simulator.play_game(
                        self.model, baseline_model,
                        self.config.max_moves_per_game,
                        self.config.time_limit_per_move
                    )
                    if game_result['winner'] == 1:
                        wins += 1
                    elif game_result['winner'] == 2:
                        losses += 1
                    else:
                        draws += 1
                else:
                    game_result = self.game_simulator.play_game(
                        baseline_model, self.model,
                        self.config.max_moves_per_game,
                        self.config.time_limit_per_move
                    )
                    if game_result['winner'] == 2:
                        wins += 1
                    elif game_result['winner'] == 1:
                        losses += 1
                    else:
                        draws += 1
                
                games_data.append(game_result)
                
                # 更新ELO评分
                score = 1 if wins > losses else (0.5 if wins == losses else 0)
                self.elo_system.update_ratings('test_model', baseline_name, score)
            
            # 计算统计
            total_games = wins + losses + draws
            win_rate = wins / total_games if total_games > 0 else 0
            
            battle_results[baseline_name] = {
                'wins': wins,
                'losses': losses, 
                'draws': draws,
                'win_rate': win_rate,
                'total_games': total_games,
                'elo_rating': self.elo_system.get_rating('test_model'),
                'avg_game_length': np.mean([g['total_moves'] for g in games_data]),
                'avg_game_time': np.mean([g['game_length'] for g in games_data]),
                'timeout_rate': sum(1 for g in games_data if g['timeout']) / len(games_data)
            }
            
            self.logger.info(f"对战 {baseline_name}: 胜率 {win_rate:.3f}, "
                           f"ELO {self.elo_system.get_rating('test_model')}")
        
        return battle_results
    
    def _create_random_model(self) -> GomokuNetwork:
        """创建随机决策模型"""
        # 创建一个简单的模型作为随机基线
        config = ModelConfig(
            board_size=15,
            input_channels=12,
            residual_blocks=2,
            filters=64
        )
        return GomokuNetwork(config)
    
    def _create_simple_model(self) -> GomokuNetwork:
        """创建简单规则模型"""
        # 创建一个简单的模型作为规则基线
        config = ModelConfig(
            board_size=15,
            input_channels=12,
            residual_blocks=4,
            filters=128
        )
        return GomokuNetwork(config)
    
    def run_full_benchmark(self) -> Dict:
        """运行完整基准测试"""
        self.logger.info("开始完整基准测试...")
        
        # 推理速度测试
        self.results['inference_speed'] = self.benchmark_inference_speed()
        
        # 内存使用测试
        self.results['memory_usage'] = self.benchmark_memory_usage()
        
        # 决策质量测试
        self.results['decision_quality'] = self.benchmark_decision_quality()
        
        # 对战测试
        self.results['battle_results'] = self.benchmark_vs_baselines()
        
        # 模型信息
        self.results['model_info'] = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024,
            'device': str(self.device),
            'config': asdict(self.model.config)
        }
        
        # 保存结果
        if self.config.save_results:
            self._save_results()
        
        # 生成图表
        if self.config.plot_results:
            self._plot_results()
        
        self.logger.info("完整基准测试完成")
        return self.results
    
    def _save_results(self):
        """保存测试结果"""
        results_file = Path(self.config.results_dir) / "benchmark_results.json"
        
        # 处理不可序列化的对象
        serializable_results = json.loads(json.dumps(self.results, default=str))
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {results_file}")
    
    def _plot_results(self):
        """绘制结果图表"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('五子棋AI模型基准测试结果', fontsize=16, fontweight='bold')
        
        # 推理时间分布
        if 'inference_speed' in self.results:
            inference_times = self.profiler.inference_times
            if inference_times:
                axes[0, 0].hist(inference_times, bins=30, alpha=0.7, color='skyblue')
                axes[0, 0].set_title('推理时间分布')
                axes[0, 0].set_xlabel('推理时间 (秒)')
                axes[0, 0].set_ylabel('频次')
        
        # 内存使用
        if 'memory_usage' in self.results:
            memory_data = self.results['memory_usage']
            batch_sizes = []
            memory_per_sample = []
            
            for bs, data in memory_data.items():
                if 'memory_per_sample' in data:
                    batch_sizes.append(int(bs))
                    memory_per_sample.append(data['memory_per_sample'] / 1024 / 1024)  # MB
            
            if batch_sizes:
                axes[0, 1].plot(batch_sizes, memory_per_sample, 'o-', color='orange')
                axes[0, 1].set_title('内存使用vs批次大小')
                axes[0, 1].set_xlabel('批次大小')
                axes[0, 1].set_ylabel('每样本内存 (MB)')
        
        # 对战胜率
        if 'battle_results' in self.results:
            battle_data = self.results['battle_results']
            opponents = list(battle_data.keys())
            win_rates = [battle_data[opp]['win_rate'] for opp in opponents]
            
            bars = axes[1, 0].bar(opponents, win_rates, color='lightgreen')
            axes[1, 0].set_title('对战胜率')
            axes[1, 0].set_ylabel('胜率')
            axes[1, 0].set_ylim(0, 1)
            
            # 添加数值标签
            for bar, rate in zip(bars, win_rates):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{rate:.3f}', ha='center', va='bottom')
        
        # 决策质量
        if 'decision_quality' in self.results and self.results['decision_quality']:
            quality_data = self.results['decision_quality']['summary']
            metrics = ['expert_top1_rate', 'expert_top3_rate', 'expert_top5_rate']
            values = [quality_data.get(metric, 0) for metric in metrics]
            labels = ['Top-1', 'Top-3', 'Top-5']
            
            axes[1, 1].bar(labels, values, color='lightcoral')
            axes[1, 1].set_title('专家移动命中率')
            axes[1, 1].set_ylabel('命中率')
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 保存图表
        if self.config.save_results:
            plot_file = Path(self.config.results_dir) / "benchmark_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"图表已保存到: {plot_file}")
        
        plt.show()

# 使用示例
def main():
    """主函数"""
    # 创建模型
    model_config = ModelConfig(
        board_size=15,
        input_channels=12,
        residual_blocks=6,
        filters=128
    )
    
    model = GomokuNetwork(model_config)
    
    # 创建基准测试配置
    benchmark_config = BenchmarkConfig(
        num_games=50,
        inference_iterations=500,
        save_results=True,
        plot_results=True
    )
    
    # 运行基准测试
    benchmark = ModelBenchmark(model, benchmark_config)
    results = benchmark.run_full_benchmark()
    
    # 输出摘要
    print("\n=== 基准测试摘要 ===")
    
    if 'model_info' in results:
        print(f"模型参数数量: {results['model_info']['total_parameters']:,}")
        print(f"模型大小: {results['model_info']['model_size_mb']:.2f} MB")
    
    if 'inference_speed' in results:
        fps = results['inference_speed']['inferences_per_second']
        print(f"推理速度: {fps:.2f} inferences/sec")
    
    if 'battle_results' in results:
        for opponent, battle_result in results['battle_results'].items():
            print(f"对战{opponent}: 胜率 {battle_result['win_rate']:.3f}")
    
    if 'decision_quality' in results and results['decision_quality']:
        quality = results['decision_quality']['summary']
        print(f"专家一致性: {quality['avg_expert_consistency']:.3f}")
        print(f"Top-1命中率: {quality['expert_top1_rate']:.3f}")

if __name__ == "__main__":
    main()