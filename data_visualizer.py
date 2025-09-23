"""
数据可视化工具
提供高级数据可视化功能

作者：Claude AI Engineer
日期：2025-09-22
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from training_data_collector import GameRecord, GameResult, BOARD_SIZE

class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        初始化可视化器
        
        Args:
            style: 绘图样式
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.color_palette = sns.color_palette("husl", 12)
    
    def plot_game_progression(self, games: List[GameRecord], 
                            output_file: str = "game_progression.html"):
        """
        绘制游戏进程图（交互式）
        
        Args:
            games: 游戏记录列表
            output_file: 输出文件
        """
        if not games:
            return
        
        # 准备数据
        game_data = []
        for i, game in enumerate(games):
            game_data.append({
                'game_index': i + 1,
                'game_length': game.game_length,
                'total_time': game.total_time,
                'result': game.final_result.name,
                'avg_thinking_time': np.mean([move.thinking_time for move in game.moves]),
                'avg_value': np.mean([move.value_estimate for move in game.moves])
            })
        
        df = pd.DataFrame(game_data)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('游戏长度变化', '平均思考时间', '平均价值估计', '游戏结果分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # 游戏长度变化
        fig.add_trace(
            go.Scatter(x=df['game_index'], y=df['game_length'], 
                      mode='lines+markers', name='游戏长度',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # 平均思考时间
        fig.add_trace(
            go.Scatter(x=df['game_index'], y=df['avg_thinking_time'], 
                      mode='lines+markers', name='平均思考时间',
                      line=dict(color='green')),
            row=1, col=2
        )
        
        # 平均价值估计
        fig.add_trace(
            go.Scatter(x=df['game_index'], y=df['avg_value'], 
                      mode='lines+markers', name='平均价值估计',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # 游戏结果分布
        result_counts = df['result'].value_counts()
        fig.add_trace(
            go.Pie(labels=result_counts.index, values=result_counts.values,
                   name="游戏结果"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="游戏进程分析",
            showlegend=True,
            height=800
        )
        
        fig.write_html(output_file)
        print(f"✅ 游戏进程图已保存: {output_file}")
    
    def plot_position_analysis(self, games: List[GameRecord], 
                             output_dir: str = "position_analysis"):
        """
        绘制位置分析图
        
        Args:
            games: 游戏记录列表
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 位置热力图（按游戏阶段）
        opening_positions = np.zeros((BOARD_SIZE, BOARD_SIZE))
        middle_positions = np.zeros((BOARD_SIZE, BOARD_SIZE))
        endgame_positions = np.zeros((BOARD_SIZE, BOARD_SIZE))
        
        for game in games:
            total_moves = len(game.moves)
            for i, move in enumerate(game.moves):
                row, col = move.position
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                    if i < total_moves * 0.3:  # 开局
                        opening_positions[row, col] += 1
                    elif i < total_moves * 0.7:  # 中局
                        middle_positions[row, col] += 1
                    else:  # 残局
                        endgame_positions[row, col] += 1
        
        # 绘制热力图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sns.heatmap(opening_positions, ax=axes[0], cmap='Blues', 
                   cbar_kws={'label': '移动次数'})
        axes[0].set_title('开局位置分布')
        
        sns.heatmap(middle_positions, ax=axes[1], cmap='Greens',
                   cbar_kws={'label': '移动次数'})
        axes[1].set_title('中局位置分布')
        
        sns.heatmap(endgame_positions, ax=axes[2], cmap='Reds',
                   cbar_kws={'label': '移动次数'})
        axes[2].set_title('残局位置分布')
        
        plt.tight_layout()
        plt.savefig(output_path / 'position_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 价值估计热力图
        self._plot_value_heatmap(games, output_path)
        
        # 3. 概率分布可视化
        self._plot_probability_distribution(games, output_path)
        
        print(f"✅ 位置分析图已保存到: {output_path}")
    
    def plot_mcts_analysis(self, games: List[GameRecord], 
                          output_file: str = "mcts_analysis.html"):
        """
        绘制MCTS分析图（交互式）
        
        Args:
            games: 游戏记录列表
            output_file: 输出文件
        """
        # 收集MCTS数据
        mcts_data = []
        for game_idx, game in enumerate(games):
            for move_idx, move in enumerate(game.moves):
                if move.mcts_probabilities and move.visit_counts:
                    probs = list(move.mcts_probabilities.values())
                    visits = list(move.visit_counts.values())
                    
                    if probs and visits:
                        # 计算熵
                        probs_array = np.array(probs)
                        probs_array = probs_array / probs_array.sum()
                        entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))
                        
                        mcts_data.append({
                            'game_index': game_idx + 1,
                            'move_index': move_idx + 1,
                            'entropy': entropy,
                            'max_prob': max(probs),
                            'total_visits': sum(visits),
                            'num_candidates': len(probs),
                            'value_estimate': move.value_estimate,
                            'thinking_time': move.thinking_time,
                            'game_stage': 'opening' if move_idx < game.game_length * 0.3 
                                         else 'middle' if move_idx < game.game_length * 0.7 
                                         else 'endgame'
                        })
        
        if not mcts_data:
            print("❌ 没有MCTS数据可分析")
            return
        
        df = pd.DataFrame(mcts_data)
        
        # 创建交互式图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('熵与最大概率关系', '访问次数分布', '价值估计vs思考时间', '游戏阶段分析'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 熵与最大概率关系
        fig.add_trace(
            go.Scatter(x=df['entropy'], y=df['max_prob'], 
                      mode='markers', name='熵vs最大概率',
                      marker=dict(color=df['total_visits'], 
                                 colorscale='Viridis', 
                                 colorbar=dict(title="总访问次数")),
                      text=[f"游戏{g}步{m}" for g, m in zip(df['game_index'], df['move_index'])],
                      hovertemplate="<b>熵:</b> %{x:.3f}<br><b>最大概率:</b> %{y:.3f}<br><b>%{text}</b>"),
            row=1, col=1
        )
        
        # 2. 访问次数分布
        fig.add_trace(
            go.Histogram(x=df['total_visits'], name='访问次数分布',
                        marker_color='lightblue'),
            row=1, col=2
        )
        
        # 3. 价值估计vs思考时间
        fig.add_trace(
            go.Scatter(x=df['thinking_time'], y=df['value_estimate'], 
                      mode='markers', name='价值vs思考时间',
                      marker=dict(color=df['entropy'], colorscale='RdYlBu'),
                      hovertemplate="<b>思考时间:</b> %{x:.3f}s<br><b>价值估计:</b> %{y:.3f}"),
            row=2, col=1
        )
        
        # 4. 游戏阶段分析
        stage_entropy = df.groupby('game_stage')['entropy'].mean()
        fig.add_trace(
            go.Bar(x=stage_entropy.index, y=stage_entropy.values, 
                   name='各阶段平均熵',
                   marker_color=['red', 'green', 'blue']),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="MCTS搜索质量分析",
            showlegend=True,
            height=800
        )
        
        fig.write_html(output_file)
        print(f"✅ MCTS分析图已保存: {output_file}")
    
    def plot_learning_curves(self, training_logs: List[Dict], 
                            output_file: str = "learning_curves.html"):
        """
        绘制学习曲线
        
        Args:
            training_logs: 训练日志
            output_file: 输出文件
        """
        if not training_logs:
            print("❌ 没有训练日志数据")
            return
        
        df = pd.DataFrame(training_logs)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('损失函数', 'AI胜率', '平均游戏长度', '模型性能'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # 1. 损失函数
        if 'policy_loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['policy_loss'], 
                          name='策略损失', line=dict(color='red')),
                row=1, col=1
            )
        if 'value_loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['value_loss'], 
                          name='价值损失', line=dict(color='blue')),
                row=1, col=1, secondary_y=True
            )
        
        # 2. AI胜率
        if 'win_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['win_rate'], 
                          name='胜率', line=dict(color='green')),
                row=1, col=2
            )
        
        # 3. 平均游戏长度
        if 'avg_game_length' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['avg_game_length'], 
                          name='平均游戏长度', line=dict(color='purple')),
                row=2, col=1
            )
        
        # 4. 模型性能
        if 'accuracy' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['accuracy'], 
                          name='准确率', line=dict(color='orange')),
                row=2, col=2
            )
        if 'mse' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['mse'], 
                          name='均方误差', line=dict(color='pink')),
                row=2, col=2, secondary_y=True
            )
        
        fig.update_layout(
            title_text="训练过程学习曲线",
            showlegend=True,
            height=800
        )
        
        fig.write_html(output_file)
        print(f"✅ 学习曲线已保存: {output_file}")
    
    def create_interactive_dashboard(self, games: List[GameRecord], 
                                   output_file: str = "dashboard.html"):
        """
        创建交互式仪表板
        
        Args:
            games: 游戏记录列表
            output_file: 输出文件
        """
        # 准备汇总数据
        summary_data = self._prepare_summary_data(games)
        
        # 创建仪表板
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '游戏结果分布', '游戏长度分布', '胜率趋势',
                '位置热力图', '价值估计分布', 'MCTS质量',
                '思考时间分布', '开局统计', '性能指标'
            ),
            specs=[
                [{"type": "pie"}, {"type": "histogram"}, {"secondary_y": False}],
                [{"type": "heatmap"}, {"type": "histogram"}, {"secondary_y": False}],
                [{"type": "histogram"}, {"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # 添加各种图表...
        self._add_dashboard_charts(fig, summary_data)
        
        fig.update_layout(
            title_text="五子棋训练数据分析仪表板",
            showlegend=True,
            height=1200
        )
        
        fig.write_html(output_file)
        print(f"✅ 交互式仪表板已保存: {output_file}")
    
    def _plot_value_heatmap(self, games: List[GameRecord], output_path: Path):
        """绘制价值估计热力图"""
        value_map = np.zeros((BOARD_SIZE, BOARD_SIZE))
        count_map = np.zeros((BOARD_SIZE, BOARD_SIZE))
        
        for game in games:
            for move in game.moves:
                row, col = move.position
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                    value_map[row, col] += move.value_estimate
                    count_map[row, col] += 1
        
        # 计算平均价值
        avg_value_map = np.divide(value_map, count_map, 
                                 out=np.zeros_like(value_map), 
                                 where=count_map!=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_value_map, center=0, cmap='RdBu_r',
                   xticklabels=range(BOARD_SIZE), 
                   yticklabels=range(BOARD_SIZE))
        plt.title('位置平均价值估计热力图')
        plt.xlabel('列')
        plt.ylabel('行')
        plt.savefig(output_path / 'value_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_probability_distribution(self, games: List[GameRecord], output_path: Path):
        """绘制概率分布可视化"""
        # 收集所有概率数据
        all_probs = []
        for game in games:
            for move in game.moves:
                if move.mcts_probabilities:
                    probs = list(move.mcts_probabilities.values())
                    all_probs.extend(probs)
        
        if not all_probs:
            return
        
        plt.figure(figsize=(12, 4))
        
        # 概率分布直方图
        plt.subplot(1, 3, 1)
        plt.hist(all_probs, bins=50, alpha=0.7, edgecolor='black')
        plt.title('MCTS概率分布')
        plt.xlabel('概率值')
        plt.ylabel('频次')
        plt.yscale('log')
        
        # 累积分布
        plt.subplot(1, 3, 2)
        sorted_probs = np.sort(all_probs)
        cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        plt.plot(sorted_probs, cumulative, 'b-', linewidth=2)
        plt.title('概率累积分布')
        plt.xlabel('概率值')
        plt.ylabel('累积概率')
        plt.grid(True, alpha=0.3)
        
        # 箱线图
        plt.subplot(1, 3, 3)
        plt.boxplot(all_probs)
        plt.title('概率分布箱线图')
        plt.ylabel('概率值')
        
        plt.tight_layout()
        plt.savefig(output_path / 'probability_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _prepare_summary_data(self, games: List[GameRecord]) -> Dict:
        """准备汇总数据"""
        # 实现数据汇总逻辑
        return {
            'total_games': len(games),
            'avg_game_length': np.mean([game.game_length for game in games]),
            'win_rates': {
                'black': sum(1 for game in games if game.final_result == GameResult.BLACK_WIN) / len(games),
                'white': sum(1 for game in games if game.final_result == GameResult.WHITE_WIN) / len(games),
                'draw': sum(1 for game in games if game.final_result == GameResult.DRAW) / len(games)
            }
        }
    
    def _add_dashboard_charts(self, fig, summary_data: Dict):
        """添加仪表板图表"""
        # 实现具体的图表添加逻辑
        pass

def test_visualizer():
    """测试可视化器"""
    print("测试数据可视化器...")
    
    # 创建可视化器
    visualizer = DataVisualizer()
    
    # 这里可以添加测试代码
    print("可视化器测试完成!")

if __name__ == "__main__":
    test_visualizer()