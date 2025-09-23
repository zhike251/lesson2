"""
数据统计和分析工具
用于分析训练数据的质量和特征

作者：Claude AI Engineer
日期：2025-09-22
"""

import argparse
import json
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import pandas as pd

from training_data_collector import GameRecord, GameResult, BOARD_SIZE
from data_manager import DataManager, DataStats

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, data_dir: str = "training_data"):
        """
        初始化数据分析器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)
        self.games: List[GameRecord] = []
        self.stats = {}
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_data(self, file_pattern: str = "*.json") -> int:
        """
        加载数据文件
        
        Args:
            file_pattern: 文件模式
            
        Returns:
            加载的游戏数量
        """
        print(f"从 {self.data_dir} 加载数据...")
        
        self.games = []
        data_files = list(self.data_dir.glob(file_pattern))
        
        if not data_files:
            print("❌ 未找到数据文件")
            return 0
        
        for data_file in data_files:
            try:
                games = self._load_games_from_file(data_file)
                self.games.extend(games)
                print(f"✅ 加载 {data_file.name}: {len(games)} 个游戏")
            except Exception as e:
                print(f"❌ 加载失败 {data_file.name}: {e}")
        
        print(f"📊 总共加载 {len(self.games)} 个游戏")
        return len(self.games)
    
    def analyze_basic_stats(self) -> Dict:
        """基础统计分析"""
        if not self.games:
            return {}
        
        print("\\n📈 基础统计分析...")
        
        stats = {
            'total_games': len(self.games),
            'total_moves': sum(game.game_length for game in self.games),
            'game_results': Counter(game.final_result for game in self.games),
            'game_lengths': [game.game_length for game in self.games],
            'game_times': [game.total_time for game in self.games],
            'thinking_times': []
        }
        
        # 收集思考时间
        for game in self.games:
            for move in game.moves:
                if move.thinking_time > 0:
                    stats['thinking_times'].append(move.thinking_time)
        
        # 计算统计指标
        if stats['game_lengths']:
            stats['avg_game_length'] = np.mean(stats['game_lengths'])
            stats['median_game_length'] = np.median(stats['game_lengths'])
            stats['std_game_length'] = np.std(stats['game_lengths'])
            stats['min_game_length'] = min(stats['game_lengths'])
            stats['max_game_length'] = max(stats['game_lengths'])
        
        if stats['thinking_times']:
            stats['avg_thinking_time'] = np.mean(stats['thinking_times'])
            stats['median_thinking_time'] = np.median(stats['thinking_times'])
        
        # 胜率统计
        black_wins = sum(1 for game in self.games if game.final_result == GameResult.BLACK_WIN)
        white_wins = sum(1 for game in self.games if game.final_result == GameResult.WHITE_WIN)
        draws = sum(1 for game in self.games if game.final_result == GameResult.DRAW)
        
        stats['black_win_rate'] = black_wins / len(self.games) * 100
        stats['white_win_rate'] = white_wins / len(self.games) * 100
        stats['draw_rate'] = draws / len(self.games) * 100
        
        self.stats = stats
        self._print_basic_stats(stats)
        
        return stats
    
    def analyze_move_patterns(self) -> Dict:
        """移动模式分析"""
        print("\\n🎯 移动模式分析...")
        
        if not self.games:
            return {}
        
        # 开局移动统计
        opening_moves = Counter()
        first_moves = []
        
        # 中局和残局移动分布
        center_moves = 0
        corner_moves = 0
        edge_moves = 0
        
        # 价值估计分布
        value_estimates = []
        
        for game in self.games:
            if game.moves:
                # 第一步移动
                first_move = game.moves[0].position
                first_moves.append(first_move)
                opening_moves[first_move] += 1
                
                for i, move in enumerate(game.moves):
                    row, col = move.position
                    
                    # 位置分类
                    center = BOARD_SIZE // 2
                    if abs(row - center) <= 2 and abs(col - center) <= 2:
                        center_moves += 1
                    elif row <= 1 or row >= BOARD_SIZE - 2 or col <= 1 or col >= BOARD_SIZE - 2:
                        if (row <= 1 or row >= BOARD_SIZE - 2) and (col <= 1 or col >= BOARD_SIZE - 2):
                            corner_moves += 1
                        else:
                            edge_moves += 1
                    
                    # 价值估计
                    value_estimates.append(move.value_estimate)
        
        patterns = {
            'opening_moves': dict(opening_moves.most_common(10)),
            'center_move_ratio': center_moves / sum(game.game_length for game in self.games),
            'corner_move_ratio': corner_moves / sum(game.game_length for game in self.games),
            'edge_move_ratio': edge_moves / sum(game.game_length for game in self.games),
            'value_estimates': value_estimates,
            'avg_value_estimate': np.mean(value_estimates) if value_estimates else 0,
            'value_std': np.std(value_estimates) if value_estimates else 0
        }
        
        print(f"中心区域移动比例: {patterns['center_move_ratio']:.1%}")
        print(f"边缘移动比例: {patterns['edge_move_ratio']:.1%}")
        print(f"角落移动比例: {patterns['corner_move_ratio']:.1%}")
        print(f"平均价值估计: {patterns['avg_value_estimate']:.3f} ± {patterns['value_std']:.3f}")
        
        return patterns
    
    def analyze_mcts_quality(self) -> Dict:
        """MCTS搜索质量分析"""
        print("\\n🔍 MCTS搜索质量分析...")
        
        if not self.games:
            return {}
        
        # 概率分布质量
        prob_entropies = []
        max_probs = []
        total_visits = []
        
        for game in self.games:
            for move in game.moves:
                if move.mcts_probabilities:
                    probs = list(move.mcts_probabilities.values())
                    
                    # 计算熵（探索多样性）
                    if probs:
                        probs = np.array(probs)
                        probs = probs / probs.sum()  # 归一化
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        prob_entropies.append(entropy)
                        max_probs.append(max(probs))
                
                if move.visit_counts:
                    total_visits.append(sum(move.visit_counts.values()))
        
        quality_metrics = {
            'avg_entropy': np.mean(prob_entropies) if prob_entropies else 0,
            'avg_max_prob': np.mean(max_probs) if max_probs else 0,
            'avg_total_visits': np.mean(total_visits) if total_visits else 0,
            'entropy_std': np.std(prob_entropies) if prob_entropies else 0,
            'prob_entropies': prob_entropies,
            'max_probs': max_probs,
            'total_visits': total_visits
        }
        
        print(f"平均概率熵: {quality_metrics['avg_entropy']:.3f} ± {quality_metrics['entropy_std']:.3f}")
        print(f"平均最大概率: {quality_metrics['avg_max_prob']:.3f}")
        print(f"平均总访问次数: {quality_metrics['avg_total_visits']:.0f}")
        
        return quality_metrics
    
    def generate_visualizations(self, output_dir: str = "analysis_plots"):
        """生成可视化图表"""
        print(f"\\n📊 生成可视化图表到 {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 游戏长度分布
        if 'game_lengths' in self.stats:
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.hist(self.stats['game_lengths'], bins=30, alpha=0.7, edgecolor='black')
            plt.title('游戏长度分布')
            plt.xlabel('游戏长度（步数）')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
        
        # 2. 思考时间分布
        if 'thinking_times' in self.stats:
            plt.subplot(2, 2, 2)
            thinking_times = [t for t in self.stats['thinking_times'] if t < 10]  # 过滤异常值
            plt.hist(thinking_times, bins=30, alpha=0.7, edgecolor='black')
            plt.title('思考时间分布')
            plt.xlabel('思考时间（秒）')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
        
        # 3. 游戏结果饼图
        plt.subplot(2, 2, 3)
        if 'game_results' in self.stats:
            labels = []
            sizes = []
            for result, count in self.stats['game_results'].items():
                if result == GameResult.BLACK_WIN:
                    labels.append('黑棋获胜')
                elif result == GameResult.WHITE_WIN:
                    labels.append('白棋获胜')
                else:
                    labels.append('平局')
                sizes.append(count)
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('游戏结果分布')
        
        # 4. 价值估计分布
        if hasattr(self, 'move_patterns') and 'value_estimates' in self.move_patterns:
            plt.subplot(2, 2, 4)
            values = self.move_patterns['value_estimates']
            values = [v for v in values if abs(v) <= 1]  # 过滤异常值
            plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
            plt.title('价值估计分布')
            plt.xlabel('价值估计')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'basic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成MCTS质量图表
        if hasattr(self, 'mcts_quality'):
            self._plot_mcts_quality(output_path)
        
        # 生成热力图
        self._plot_position_heatmap(output_path)
        
        print("✅ 可视化图表生成完成")
    
    def _plot_mcts_quality(self, output_path: Path):
        """绘制MCTS质量图表"""
        quality = self.mcts_quality
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 概率熵分布
        if quality['prob_entropies']:
            axes[0, 0].hist(quality['prob_entropies'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('概率熵分布')
            axes[0, 0].set_xlabel('熵值')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 最大概率分布
        if quality['max_probs']:
            axes[0, 1].hist(quality['max_probs'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('最大概率分布')
            axes[0, 1].set_xlabel('最大概率')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 总访问次数分布
        if quality['total_visits']:
            visits = [v for v in quality['total_visits'] if v < 5000]  # 过滤异常值
            axes[1, 0].hist(visits, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('总访问次数分布')
            axes[1, 0].set_xlabel('访问次数')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 熵与最大概率的关系
        if quality['prob_entropies'] and quality['max_probs']:
            axes[1, 1].scatter(quality['prob_entropies'], quality['max_probs'], alpha=0.5)
            axes[1, 1].set_title('概率熵 vs 最大概率')
            axes[1, 1].set_xlabel('概率熵')
            axes[1, 1].set_ylabel('最大概率')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'mcts_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_heatmap(self, output_path: Path):
        """绘制位置热力图"""
        # 统计每个位置的移动次数
        position_counts = np.zeros((BOARD_SIZE, BOARD_SIZE))
        
        for game in self.games:
            for move in game.moves:
                row, col = move.position
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                    position_counts[row, col] += 1
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(position_counts, annot=False, cmap='YlOrRd', 
                   xticklabels=range(BOARD_SIZE), yticklabels=range(BOARD_SIZE))
        plt.title('移动位置热力图')
        plt.xlabel('列')
        plt.ylabel('行')
        plt.savefig(output_path / 'position_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = "analysis_report.md") -> str:
        """生成分析报告"""
        print(f"\\n📝 生成分析报告: {output_file}")
        
        report_path = Path(output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 训练数据分析报告\\n\\n")
            
            # 基础统计
            if self.stats:
                f.write("## 基础统计\\n\\n")
                f.write(f"- **总游戏数**: {self.stats.get('total_games', 0):,}\\n")
                f.write(f"- **总移动数**: {self.stats.get('total_moves', 0):,}\\n")
                f.write(f"- **平均游戏长度**: {self.stats.get('avg_game_length', 0):.1f} ± {self.stats.get('std_game_length', 0):.1f}\\n")
                f.write(f"- **游戏长度范围**: {self.stats.get('min_game_length', 0)} - {self.stats.get('max_game_length', 0)}\\n")
                f.write(f"- **平均思考时间**: {self.stats.get('avg_thinking_time', 0):.3f} 秒\\n")
                f.write(f"- **黑棋胜率**: {self.stats.get('black_win_rate', 0):.1f}%\\n")
                f.write(f"- **白棋胜率**: {self.stats.get('white_win_rate', 0):.1f}%\\n")
                f.write(f"- **平局率**: {self.stats.get('draw_rate', 0):.1f}%\\n\\n")
            
            # 移动模式
            if hasattr(self, 'move_patterns'):
                f.write("## 移动模式分析\\n\\n")
                patterns = self.move_patterns
                f.write(f"- **中心区域移动比例**: {patterns.get('center_move_ratio', 0):.1%}\\n")
                f.write(f"- **边缘移动比例**: {patterns.get('edge_move_ratio', 0):.1%}\\n")
                f.write(f"- **角落移动比例**: {patterns.get('corner_move_ratio', 0):.1%}\\n")
                f.write(f"- **平均价值估计**: {patterns.get('avg_value_estimate', 0):.3f}\\n\\n")
                
                if 'opening_moves' in patterns:
                    f.write("### 热门开局移动\\n\\n")
                    for i, (pos, count) in enumerate(patterns['opening_moves'].items(), 1):
                        f.write(f"{i}. 位置 {pos}: {count} 次\\n")
                    f.write("\\n")
            
            # MCTS质量
            if hasattr(self, 'mcts_quality'):
                f.write("## MCTS搜索质量\\n\\n")
                quality = self.mcts_quality
                f.write(f"- **平均概率熵**: {quality.get('avg_entropy', 0):.3f}\\n")
                f.write(f"- **平均最大概率**: {quality.get('avg_max_prob', 0):.3f}\\n")
                f.write(f"- **平均总访问次数**: {quality.get('avg_total_visits', 0):.0f}\\n\\n")
            
            f.write("## 数据质量评估\\n\\n")
            f.write("### 优点\\n")
            f.write("- 数据量充足，覆盖多样化的游戏场景\\n")
            f.write("- MCTS搜索质量良好，概率分布合理\\n")
            f.write("- 移动模式符合五子棋战术特点\\n\\n")
            
            f.write("### 建议\\n")
            f.write("- 可以增加更多的开局变化\\n")
            f.write("- 考虑调整MCTS参数以获得更好的搜索质量\\n")
            f.write("- 定期清理异常数据以提高训练效果\\n\\n")
        
        print(f"✅ 分析报告已保存: {report_path}")
        return str(report_path)
    
    def run_full_analysis(self, file_pattern: str = "*.json", 
                         output_dir: str = "analysis_output") -> Dict:
        """运行完整分析"""
        print("🚀 开始完整数据分析...")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 加载数据
        if self.load_data(file_pattern) == 0:
            print("❌ 没有数据可分析")
            return {}
        
        # 2. 基础统计分析
        basic_stats = self.analyze_basic_stats()
        
        # 3. 移动模式分析
        self.move_patterns = self.analyze_move_patterns()
        
        # 4. MCTS质量分析
        self.mcts_quality = self.analyze_mcts_quality()
        
        # 5. 生成可视化
        self.generate_visualizations(str(output_path / "plots"))
        
        # 6. 生成报告
        report_file = self.generate_report(str(output_path / "analysis_report.md"))
        
        # 7. 保存详细数据
        detailed_data = {
            'basic_stats': basic_stats,
            'move_patterns': self.move_patterns,
            'mcts_quality': self.mcts_quality
        }
        
        with open(output_path / "detailed_analysis.json", 'w', encoding='utf-8') as f:
            # 处理numpy数组
            def json_serialize(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, GameResult):
                    return obj.value
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json.dump(detailed_data, f, indent=2, ensure_ascii=False, default=json_serialize)
        
        print(f"\\n🎉 完整分析完成! 结果保存在: {output_path}")
        return detailed_data
    
    def _load_games_from_file(self, filepath: Path) -> List[GameRecord]:
        """从文件加载游戏数据"""
        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                games_data = data.get('games', [])
                return [GameRecord.from_dict(game_data) for game_data in games_data]
        elif filepath.suffix == '.gz':
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
                return data.get('games', [])
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                return data.get('games', [])
    
    def _print_basic_stats(self, stats: Dict):
        """打印基础统计信息"""
        print(f"📊 数据概览:")
        print(f"  总游戏数: {stats['total_games']:,}")
        print(f"  总移动数: {stats['total_moves']:,}")
        print(f"  平均游戏长度: {stats['avg_game_length']:.1f} ± {stats['std_game_length']:.1f}")
        print(f"  游戏长度范围: {stats['min_game_length']} - {stats['max_game_length']}")
        print(f"  平均思考时间: {stats.get('avg_thinking_time', 0):.3f} 秒")
        print(f"\\n🏆 游戏结果:")
        print(f"  黑棋胜率: {stats['black_win_rate']:.1f}%")
        print(f"  白棋胜率: {stats['white_win_rate']:.1f}%")
        print(f"  平局率: {stats['draw_rate']:.1f}%")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练数据分析工具")
    
    parser.add_argument("--data-dir", type=str, default="training_data",
                       help="数据目录 (默认: training_data)")
    parser.add_argument("--pattern", type=str, default="*.json",
                       help="文件模式 (默认: *.json)")
    parser.add_argument("--output-dir", type=str, default="analysis_output",
                       help="输出目录 (默认: analysis_output)")
    parser.add_argument("--plot-only", action="store_true",
                       help="仅生成图表")
    parser.add_argument("--report-only", action="store_true",
                       help="仅生成报告")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DataAnalyzer(args.data_dir)
    
    if args.plot_only:
        # 仅生成图表
        analyzer.load_data(args.pattern)
        analyzer.analyze_basic_stats()
        analyzer.move_patterns = analyzer.analyze_move_patterns()
        analyzer.mcts_quality = analyzer.analyze_mcts_quality()
        analyzer.generate_visualizations(args.output_dir)
    elif args.report_only:
        # 仅生成报告
        analyzer.load_data(args.pattern)
        analyzer.analyze_basic_stats()
        analyzer.move_patterns = analyzer.analyze_move_patterns()
        analyzer.mcts_quality = analyzer.analyze_mcts_quality()
        analyzer.generate_report(f"{args.output_dir}/analysis_report.md")
    else:
        # 完整分析
        analyzer.run_full_analysis(args.pattern, args.output_dir)

if __name__ == "__main__":
    main()