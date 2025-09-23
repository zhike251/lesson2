"""
自动化自我对弈脚本
用于批量生成训练数据

作者：Claude AI Engineer
日期：2025-09-22
"""

import argparse
import time
import signal
import sys
import os
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from self_play_engine import SelfPlayEngine, SelfPlayConfig, SelfPlayMode
from training_data_collector import TrainingDataCollector
from neural_mcts import NeuralMCTSAdapter, AlphaZeroStyleNetwork

class AutoSelfPlay:
    """自动化自我对弈"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化自动化自我对弈
        
        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.engine = None
        self.stop_requested = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_file: Optional[str]) -> SelfPlayConfig:
        """加载配置"""
        if config_file and os.path.exists(config_file):
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            return SelfPlayConfig(**config_dict)
        else:
            # 默认配置
            return SelfPlayConfig(
                num_games=100,
                max_moves_per_game=450,
                thinking_time_limit=3.0,
                mcts_simulations=800,
                temperature=1.0,
                temperature_decay=0.95,
                min_temperature=0.1,
                use_opening_book=True,
                randomize_opening=True,
                collect_training_data=True,
                save_frequency=10,
                enable_progress_callback=True,
                parallel_games=1,
                max_threads=4
            )
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在停止自我对弈...")
        self.stop_requested = True
        if self.engine:
            self.engine.stop_self_play()
    
    def run(self, mode: str = "single", output_dir: str = "selfplay_data") -> Dict:
        """
        运行自我对弈
        
        Args:
            mode: 运行模式 ("single", "multi", "batch")
            output_dir: 输出目录
            
        Returns:
            运行结果
        """
        print("=" * 60)
        print("自动化自我对弈系统")
        print("=" * 60)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 创建数据收集器
        collector = TrainingDataCollector(
            data_dir=str(output_path),
            auto_save_frequency=self.config.save_frequency,
            max_memory_games=100
        )
        
        # 创建神经网络（使用dummy网络用于测试）
        neural_network = AlphaZeroStyleNetwork()
        
        # 创建Neural MCTS适配器
        mcts_adapter = NeuralMCTSAdapter(
            neural_network=neural_network,
            mcts_simulations=self.config.mcts_simulations,
            c_puct=1.25,
            time_limit=self.config.thinking_time_limit
        )
        
        # 启用数据收集
        mcts_adapter.set_data_collection(True)
        
        # 创建自我对弈引擎
        self.engine = SelfPlayEngine(
            neural_mcts_engine=mcts_adapter,
            config=self.config,
            data_collector=collector
        )
        
        # 添加进度回调
        self.engine.add_progress_callback(self._progress_callback)
        
        # 运行自我对弈
        try:
            play_mode = SelfPlayMode(mode)
            start_time = time.time()
            
            print(f"开始自我对弈 (模式: {mode})")
            print(f"目标游戏数: {self.config.num_games}")
            print(f"MCTS模拟次数: {self.config.mcts_simulations}")
            print(f"输出目录: {output_path}")
            print("-" * 40)
            
            results = self.engine.run_self_play(play_mode)
            
            if not self.stop_requested:
                # 保存最终数据
                final_file = self.engine.save_final_data()
                print(f"\n最终数据已保存: {final_file}")
                
                # 生成总结报告
                self._generate_report(results, output_path, time.time() - start_time)
                
                print("\n✅ 自我对弈完成!")
            else:
                print("\n⚠️ 自我对弈被中断")
                
            return results
            
        except Exception as e:
            print(f"\n❌ 自我对弈失败: {e}")
            return {}
    
    def _progress_callback(self, progress: Dict):
        """进度回调"""
        if progress['completed_games'] % 5 == 0 or progress['completed_games'] == 1:
            print(f"进度: {progress['completed_games']}/{progress['total_games']} "
                  f"({progress['progress_percent']:.1f}%) "
                  f"| 黑胜: {progress['current_stats']['black_wins']} "
                  f"| 白胜: {progress['current_stats']['white_wins']} "
                  f"| 平局: {progress['current_stats']['draws']}")
    
    def _generate_report(self, results: Dict, output_path: Path, total_time: float):
        """生成报告"""
        timestamp = int(time.time())
        report_file = output_path / f"selfplay_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("自我对弈总结报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n")
            f.write(f"总耗时: {total_time:.2f} 秒\n")
            f.write(f"完成游戏数: {results.get('games_completed', 0)}\n")
            f.write(f"总移动数: {results.get('total_moves', 0)}\n")
            f.write(f"平均游戏长度: {results.get('avg_game_length', 0):.1f}\n")
            f.write(f"平均每步时间: {results.get('avg_time_per_move', 0):.3f} 秒\n\n")
            
            f.write("游戏结果统计:\n")
            f.write(f"  黑棋获胜: {results.get('black_wins', 0)}\n")
            f.write(f"  白棋获胜: {results.get('white_wins', 0)}\n")
            f.write(f"  平局: {results.get('draws', 0)}\n\n")
            
            if results.get('black_wins', 0) + results.get('white_wins', 0) > 0:
                total_decisive = results.get('black_wins', 0) + results.get('white_wins', 0)
                black_win_rate = results.get('black_wins', 0) / total_decisive * 100
                white_win_rate = results.get('white_wins', 0) / total_decisive * 100
                f.write(f"黑棋胜率: {black_win_rate:.1f}%\n")
                f.write(f"白棋胜率: {white_win_rate:.1f}%\n\n")
            
            f.write("配置参数:\n")
            f.write(f"  MCTS模拟次数: {self.config.mcts_simulations}\n")
            f.write(f"  思考时间限制: {self.config.thinking_time_limit} 秒\n")
            f.write(f"  初始温度: {self.config.temperature}\n")
            f.write(f"  温度衰减: {self.config.temperature_decay}\n")
            f.write(f"  最小温度: {self.config.min_temperature}\n")
            f.write(f"  使用开局库: {self.config.use_opening_book}\n")
            f.write(f"  并行游戏数: {self.config.parallel_games}\n")
        
        print(f"报告已保存: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自动化自我对弈脚本")
    
    parser.add_argument("--games", type=int, default=100,
                       help="游戏数量 (默认: 100)")
    parser.add_argument("--simulations", type=int, default=800,
                       help="MCTS模拟次数 (默认: 800)")
    parser.add_argument("--time-limit", type=float, default=3.0,
                       help="思考时间限制 (默认: 3.0秒)")
    parser.add_argument("--mode", choices=["single", "multi", "batch"], 
                       default="single", help="运行模式 (默认: single)")
    parser.add_argument("--output-dir", type=str, default="selfplay_data",
                       help="输出目录 (默认: selfplay_data)")
    parser.add_argument("--config", type=str, 
                       help="配置文件路径")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="初始温度 (默认: 1.0)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="并行游戏数 (默认: 1)")
    parser.add_argument("--threads", type=int, default=4,
                       help="最大线程数 (默认: 4)")
    
    args = parser.parse_args()
    
    # 创建配置
    config = SelfPlayConfig(
        num_games=args.games,
        mcts_simulations=args.simulations,
        thinking_time_limit=args.time_limit,
        temperature=args.temperature,
        parallel_games=args.parallel,
        max_threads=args.threads,
        collect_training_data=True,
        save_frequency=max(1, args.games // 10)
    )
    
    # 创建自动化自我对弈
    auto_play = AutoSelfPlay(args.config)
    auto_play.config = config  # 覆盖配置
    
    # 运行自我对弈
    results = auto_play.run(mode=args.mode, output_dir=args.output_dir)
    
    if results:
        print("\n🎉 任务完成!")
    else:
        print("\n❌ 任务失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()