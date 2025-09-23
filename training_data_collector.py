"""
训练数据收集模块
实现自我对弈、数据收集和管理功能

作者：Claude AI Engineer  
日期：2025-09-23
"""

import json
import time
import pickle
import random
import threading
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict

# 导入现有模块
from neural_evaluator import NeuralNetworkEvaluator
from neural_mcts import NeuralMCTS
from integrated_ai import IntegratedGomokuAI, BOARD_SIZE, EMPTY, BLACK, WHITE

@dataclass
class GamePosition:
    """单个游戏位置的训练数据"""
    board_state: List[List[int]]  # 棋盘状态
    current_player: int  # 当前玩家
    move_probabilities: List[float]  # MCTS搜索得到的移动概率分布
    actual_move: Tuple[int, int]  # 实际选择的移动
    game_result: float  # 游戏结果 (1.0=胜利, 0.0=失败, 0.5=平局)
    position_value: float  # 位置价值评估
    move_number: int  # 移动序号
    search_time: float  # 搜索用时
    nodes_searched: int  # 搜索节点数

@dataclass 
class GameRecord:
    """完整游戏记录"""
    game_id: str
    positions: List[GamePosition]
    final_result: Dict[str, Any]  # 最终结果信息
    game_metadata: Dict[str, Any]  # 游戏元数据
    duration: float  # 游戏时长
    total_moves: int  # 总移动数

class TrainingDataCollector:
    """训练数据收集器"""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.game_records: List[GameRecord] = []
        self.position_buffer: List[GamePosition] = []
        
        # 统计信息
        self.stats = {
            'total_games': 0,
            'total_positions': 0,
            'win_stats': {'BLACK': 0, 'WHITE': 0, 'DRAW': 0},
            'avg_game_length': 0.0,
            'data_collection_time': 0.0
        }
        
        # 配置
        self.max_buffer_size = 10000
        self.auto_save_interval = 100  # 每100局自动保存
        self.data_augmentation = True
        
    def start_game_recording(self, game_id: str = None) -> str:
        """开始记录新游戏"""
        if game_id is None:
            game_id = f"game_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.current_game_id = game_id
        self.current_positions = []
        self.game_start_time = time.time()
        
        return game_id
    
    def record_position(self, 
                       board_state: List[List[int]], 
                       current_player: int,
                       move_probabilities: List[float],
                       actual_move: Tuple[int, int],
                       position_value: float,
                       search_time: float,
                       nodes_searched: int):
        """记录游戏位置数据"""
        
        position = GamePosition(
            board_state=self._copy_board(board_state),
            current_player=current_player,
            move_probabilities=move_probabilities.copy(),
            actual_move=actual_move,
            game_result=0.0,  # 将在游戏结束后更新
            position_value=position_value,
            move_number=len(self.current_positions),
            search_time=search_time,
            nodes_searched=nodes_searched
        )
        
        self.current_positions.append(position)
    
    def finish_game_recording(self, winner: int, metadata: Dict = None):
        """完成游戏记录"""
        if not hasattr(self, 'current_positions'):
            return
        
        game_duration = time.time() - self.game_start_time
        
        # 更新每个位置的游戏结果
        for i, position in enumerate(self.current_positions):
            if winner == 0:  # 平局
                position.game_result = 0.5
            elif position.current_player == winner:
                position.game_result = 1.0
            else:
                position.game_result = 0.0
        
        # 创建游戏记录
        final_result = {
            'winner': winner,
            'winner_name': 'BLACK' if winner == BLACK else 'WHITE' if winner == WHITE else 'DRAW',
            'total_moves': len(self.current_positions),
            'game_duration': game_duration
        }
        
        game_metadata = metadata or {}
        game_metadata.update({
            'timestamp': time.time(),
            'collection_version': '1.0'
        })
        
        game_record = GameRecord(
            game_id=self.current_game_id,
            positions=self.current_positions.copy(),
            final_result=final_result,
            game_metadata=game_metadata,
            duration=game_duration,
            total_moves=len(self.current_positions)
        )
        
        # 存储记录
        self.game_records.append(game_record)
        self.position_buffer.extend(self.current_positions)
        
        # 更新统计
        self._update_stats(game_record)
        
        # 数据增强
        if self.data_augmentation:
            self._augment_game_data(game_record)
        
        # 清理
        delattr(self, 'current_positions')
        delattr(self, 'current_game_id')
        delattr(self, 'game_start_time')
        
        # 自动保存检查
        if len(self.game_records) % self.auto_save_interval == 0:
            self.save_data()
    
    def _copy_board(self, board: List[List[int]]) -> List[List[int]]:
        """深拷贝棋盘"""
        return [row.copy() for row in board]
    
    def _update_stats(self, game_record: GameRecord):
        """更新统计信息"""
        self.stats['total_games'] += 1
        self.stats['total_positions'] += len(game_record.positions)
        
        winner_name = game_record.final_result['winner_name']
        self.stats['win_stats'][winner_name] += 1
        
        # 更新平均游戏长度
        total_moves = sum(record.total_moves for record in self.game_records)
        self.stats['avg_game_length'] = total_moves / len(self.game_records)
    
    def _augment_game_data(self, game_record: GameRecord):
        """数据增强：生成旋转和镜像变换"""
        augmented_positions = []
        
        for position in game_record.positions:
            # 生成8种对称变换
            for transform_id in range(8):
                augmented_pos = self._transform_position(position, transform_id)
                augmented_positions.append(augmented_pos)
        
        # 创建增强的游戏记录
        augmented_record = GameRecord(
            game_id=f"{game_record.game_id}_aug",
            positions=augmented_positions,
            final_result=game_record.final_result.copy(),
            game_metadata={**game_record.game_metadata, 'augmented': True},
            duration=game_record.duration,
            total_moves=len(augmented_positions)
        )
        
        self.position_buffer.extend(augmented_positions)
    
    def _transform_position(self, position: GamePosition, transform_id: int) -> GamePosition:
        """应用几何变换"""
        # 这里实现8种对称变换
        # 简化版本，实际应用中需要完整实现所有变换
        transformed_board = self._transform_board(position.board_state, transform_id)
        transformed_move = self._transform_move(position.actual_move, transform_id)
        transformed_probs = self._transform_probabilities(position.move_probabilities, transform_id)
        
        return GamePosition(
            board_state=transformed_board,
            current_player=position.current_player,
            move_probabilities=transformed_probs,
            actual_move=transformed_move,
            game_result=position.game_result,
            position_value=position.position_value,
            move_number=position.move_number,
            search_time=position.search_time,
            nodes_searched=position.nodes_searched
        )
    
    def _transform_board(self, board: List[List[int]], transform_id: int) -> List[List[int]]:
        """变换棋盘"""
        # 简化实现：这里只实现一种变换作为示例
        if transform_id == 0:
            return [row.copy() for row in board]
        elif transform_id == 1:  # 水平镜像
            return [row[::-1] for row in board]
        else:
            return [row.copy() for row in board]  # 其他变换待实现
    
    def _transform_move(self, move: Tuple[int, int], transform_id: int) -> Tuple[int, int]:
        """变换移动位置"""
        row, col = move
        if transform_id == 1:  # 水平镜像
            return (row, BOARD_SIZE - 1 - col)
        return move
    
    def _transform_probabilities(self, probs: List[float], transform_id: int) -> List[float]:
        """变换概率分布"""
        # 简化实现
        return probs.copy()
    
    def get_training_batch(self, batch_size: int = 32) -> List[GamePosition]:
        """获取训练批次"""
        if len(self.position_buffer) < batch_size:
            return self.position_buffer.copy()
        
        return random.sample(self.position_buffer, batch_size)
    
    def save_data(self, filename: str = None):
        """保存数据到文件"""
        if filename is None:
            filename = f"training_data_{int(time.time())}.pkl"
        
        filepath = self.data_dir / filename
        
        data = {
            'game_records': self.game_records,
            'position_buffer': self.position_buffer,
            'stats': self.stats,
            'metadata': {
                'save_time': time.time(),
                'version': '1.0'
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"数据已保存到: {filepath}")
        return filepath
    
    def load_data(self, filepath: str):
        """从文件加载数据"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.game_records = data['game_records']
        self.position_buffer = data['position_buffer']
        self.stats = data['stats']
        
        print(f"数据已从 {filepath} 加载")
        print(f"游戏数量: {len(self.game_records)}")
        print(f"位置数量: {len(self.position_buffer)}")
    
    def get_stats_summary(self) -> Dict:
        """获取统计摘要"""
        return {
            'collection_stats': self.stats,
            'buffer_status': {
                'total_positions': len(self.position_buffer),
                'buffer_size_mb': len(pickle.dumps(self.position_buffer)) / (1024*1024),
                'ready_for_training': len(self.position_buffer) >= 1000
            },
            'data_quality': self._assess_data_quality()
        }
    
    def _assess_data_quality(self) -> Dict:
        """评估数据质量"""
        if not self.position_buffer:
            return {'status': 'no_data'}
        
        # 分析数据分布
        game_lengths = [len(record.positions) for record in self.game_records]
        position_values = [pos.position_value for pos in self.position_buffer]
        search_times = [pos.search_time for pos in self.position_buffer]
        
        return {
            'avg_game_length': np.mean(game_lengths) if game_lengths else 0,
            'avg_position_value': np.mean(position_values) if position_values else 0,
            'avg_search_time': np.mean(search_times) if search_times else 0,
            'value_std': np.std(position_values) if position_values else 0,
            'data_coverage': len(set(tuple(map(tuple, pos.board_state)) for pos in self.position_buffer[:1000]))
        }

class SelfPlayEngine:
    """自我对弈引擎"""
    
    def __init__(self, 
                 ai_config: Dict = None,
                 data_collector: TrainingDataCollector = None,
                 num_simulations: int = 800):
        
        self.ai_config = ai_config or {'ai_difficulty': 'neural_mcts'}
        self.data_collector = data_collector or TrainingDataCollector()
        self.num_simulations = num_simulations
        
        # 创建AI系统
        self.ai_system = IntegratedGomokuAI(**self.ai_config)
        
        # 统计
        self.games_played = 0
        self.total_time = 0
        
    def play_single_game(self, save_data: bool = True, verbose: bool = False) -> GameRecord:
        """进行单局自我对弈"""
        
        # 初始化游戏
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        current_player = BLACK
        game_over = False
        winner = 0
        move_count = 0
        max_moves = BOARD_SIZE * BOARD_SIZE
        
        # 开始记录
        game_id = self.data_collector.start_game_recording()
        
        if verbose:
            print(f"开始自我对弈游戏: {game_id}")
        
        game_start = time.time()
        
        while not game_over and move_count < max_moves:
            # 获取AI移动和搜索信息
            move_start = time.time()
            
            # 使用Neural MCTS获取移动
            result = self.ai_system.ai_engine.get_best_move(board, current_player)
            
            if not result.move:
                break
                
            move_time = time.time() - move_start
            row, col = result.move
            
            # 生成概率分布（简化版，实际应从MCTS获取）
            move_probabilities = [0.0] * (BOARD_SIZE * BOARD_SIZE)
            move_idx = row * BOARD_SIZE + col
            move_probabilities[move_idx] = 1.0  # 简化版本
            
            # 记录位置数据
            if save_data:
                self.data_collector.record_position(
                    board_state=board,
                    current_player=current_player,
                    move_probabilities=move_probabilities,
                    actual_move=(row, col),
                    position_value=result.score,
                    search_time=move_time,
                    nodes_searched=result.nodes_searched
                )
            
            # 执行移动
            board[row][col] = current_player
            
            # 检查胜负
            if self._check_win(board, row, col, current_player):
                winner = current_player
                game_over = True
            else:
                current_player = 3 - current_player
                move_count += 1
            
            if verbose and move_count % 10 == 0:
                print(f"  移动 {move_count}: ({row}, {col})")
        
        game_duration = time.time() - game_start
        
        # 完成记录
        if save_data:
            metadata = {
                'self_play': True,
                'ai_config': self.ai_config,
                'num_simulations': self.num_simulations
            }
            self.data_collector.finish_game_recording(winner, metadata)
        
        # 更新统计
        self.games_played += 1
        self.total_time += game_duration
        
        if verbose:
            winner_name = 'BLACK' if winner == BLACK else 'WHITE' if winner == WHITE else 'DRAW'
            print(f"游戏结束: {winner_name} 获胜, 用时: {game_duration:.2f}秒, 移动数: {move_count}")
        
        return self.data_collector.game_records[-1] if save_data else None
    
    def _check_win(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == player:
                count += 1
                x += dx
                y += dy
            
            # 负方向  
            x, y = row - dx, col - dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            
            if count >= 5:
                return True
        
        return False
    
    def run_self_play_batch(self, 
                           num_games: int = 100,
                           parallel: bool = False,
                           verbose: bool = True) -> List[GameRecord]:
        """运行批量自我对弈"""
        
        print(f"开始批量自我对弈: {num_games} 局游戏")
        
        batch_start = time.time()
        games_records = []
        
        if parallel:
            # 并行执行（简化实现）
            games_records = self._run_parallel_games(num_games, verbose)
        else:
            # 串行执行
            for i in range(num_games):
                if verbose and (i + 1) % 10 == 0:
                    print(f"完成游戏: {i + 1}/{num_games}")
                
                record = self.play_single_game(save_data=True, verbose=False)
                if record:
                    games_records.append(record)
        
        batch_time = time.time() - batch_start
        
        if verbose:
            print(f"\n批量自我对弈完成!")
            print(f"总游戏数: {len(games_records)}")
            print(f"总用时: {batch_time:.2f}秒")
            print(f"平均每局用时: {batch_time/len(games_records):.2f}秒")
            
            # 打印统计信息
            stats = self.data_collector.get_stats_summary()
            print(f"数据收集统计: {stats['collection_stats']}")
        
        return games_records
    
    def _run_parallel_games(self, num_games: int, verbose: bool) -> List[GameRecord]:
        """并行运行游戏（简化实现）"""
        # 这里实现并行版本，当前返回串行结果
        games_records = []
        for i in range(num_games):
            record = self.play_single_game(save_data=True, verbose=False)
            if record:
                games_records.append(record)
        return games_records

class TrainingDataManager:
    """训练数据管理器"""
    
    def __init__(self, data_collector: TrainingDataCollector):
        self.data_collector = data_collector
    
    def export_to_json(self, output_file: str, limit: int = 1000):
        """导出数据到JSON格式"""
        positions = self.data_collector.position_buffer[:limit]
        
        json_data = {
            'metadata': {
                'export_time': time.time(),
                'total_positions': len(positions),
                'data_version': '1.0'
            },
            'positions': []
        }
        
        for pos in positions:
            json_data['positions'].append({
                'board': pos.board_state,
                'player': pos.current_player, 
                'move_probs': pos.move_probabilities,
                'actual_move': pos.actual_move,
                'result': pos.game_result,
                'value': pos.position_value
            })
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"数据已导出到: {output_file}")
    
    def analyze_data_distribution(self):
        """分析数据分布"""
        positions = self.data_collector.position_buffer
        
        if not positions:
            print("没有数据可分析")
            return
        
        print("=== 训练数据分析 ===")
        print(f"总位置数: {len(positions)}")
        
        # 游戏结果分布
        results = [pos.game_result for pos in positions]
        win_count = sum(1 for r in results if r > 0.5)
        loss_count = sum(1 for r in results if r < 0.5) 
        draw_count = sum(1 for r in results if r == 0.5)
        
        print(f"胜利位置: {win_count} ({win_count/len(results)*100:.1f}%)")
        print(f"失败位置: {loss_count} ({loss_count/len(results)*100:.1f}%)")
        print(f"平局位置: {draw_count} ({draw_count/len(results)*100:.1f}%)")
        
        # 价值分布
        values = [pos.position_value for pos in positions]
        print(f"位置价值范围: [{min(values):.3f}, {max(values):.3f}]")
        print(f"平均位置价值: {np.mean(values):.3f} ± {np.std(values):.3f}")
        
        # 搜索统计
        search_times = [pos.search_time for pos in positions]
        nodes_searched = [pos.nodes_searched for pos in positions]
        
        print(f"平均搜索时间: {np.mean(search_times):.3f}秒")
        print(f"平均搜索节点: {np.mean(nodes_searched):.0f}")
        
    def create_train_test_split(self, test_ratio: float = 0.2) -> Tuple[List, List]:
        """创建训练/测试数据集分割"""
        positions = self.data_collector.position_buffer.copy()
        random.shuffle(positions)
        
        split_idx = int(len(positions) * (1 - test_ratio))
        train_set = positions[:split_idx]
        test_set = positions[split_idx:]
        
        print(f"数据集分割完成:")
        print(f"训练集: {len(train_set)} 个位置")
        print(f"测试集: {len(test_set)} 个位置")
        
        return train_set, test_set

# 工具函数
def run_automated_self_play(num_games: int = 100, 
                           data_dir: str = "training_data",
                           ai_difficulty: str = "neural_mcts"):
    """运行自动化自我对弈"""
    
    print(f"=== 自动化自我对弈开始 ===")
    print(f"目标游戏数: {num_games}")
    print(f"AI难度: {ai_difficulty}")
    print(f"数据目录: {data_dir}")
    
    # 创建收集器和引擎
    collector = TrainingDataCollector(data_dir)
    engine = SelfPlayEngine(
        ai_config={'ai_difficulty': ai_difficulty},
        data_collector=collector
    )
    
    # 运行自我对弈
    records = engine.run_self_play_batch(num_games, verbose=True)
    
    # 保存数据
    data_file = collector.save_data()
    
    # 数据分析
    manager = TrainingDataManager(collector)
    manager.analyze_data_distribution()
    
    # 导出JSON
    json_file = Path(data_dir) / f"positions_{int(time.time())}.json"
    manager.export_to_json(str(json_file))
    
    print(f"\n=== 自我对弈完成 ===")
    print(f"数据文件: {data_file}")
    print(f"JSON文件: {json_file}")
    
    return collector, records

if __name__ == "__main__":
    # 测试训练数据收集
    print("测试训练数据收集系统...")
    
    # 运行少量自我对弈测试
    collector, records = run_automated_self_play(
        num_games=5,
        ai_difficulty="medium"  # 使用传统AI进行快速测试
    )
    
    print("训练数据收集系统测试完成!")