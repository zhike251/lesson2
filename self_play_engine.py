"""
自我对弈引擎
实现五子棋AI的自我对弈训练系统

作者：Claude AI Engineer
日期：2025-09-22
"""

import time
import threading
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import os

from training_data_collector import (
    TrainingDataCollector, GameRecord, GameResult, MoveData, BOARD_SIZE, EMPTY, BLACK, WHITE
)

class SelfPlayMode(Enum):
    """自我对弈模式"""
    SINGLE_THREAD = "single"
    MULTI_THREAD = "multi"
    BATCH = "batch"

@dataclass
class SelfPlayConfig:
    """自我对弈配置"""
    num_games: int = 100  # 对弈游戏数
    max_moves_per_game: int = 450  # 每局最大步数
    thinking_time_limit: float = 3.0  # 思考时间限制
    mcts_simulations: int = 800  # MCTS模拟次数
    temperature: float = 1.0  # 温度参数
    temperature_decay: float = 0.95  # 温度衰减
    min_temperature: float = 0.1  # 最小温度
    use_opening_book: bool = True  # 使用开局库
    randomize_opening: bool = True  # 随机化开局
    collect_training_data: bool = True  # 收集训练数据
    save_frequency: int = 10  # 保存频率
    enable_progress_callback: bool = True  # 启用进度回调
    parallel_games: int = 1  # 并行游戏数
    max_threads: int = 4  # 最大线程数

@dataclass
class GameState:
    """游戏状态"""
    board: np.ndarray  # 棋盘状态
    current_player: int  # 当前玩家
    move_count: int = 0  # 移动次数
    game_over: bool = False  # 游戏是否结束
    winner: int = 0  # 获胜者
    move_history: List[Tuple[int, int]] = field(default_factory=list)  # 移动历史
    
    def copy(self) -> 'GameState':
        """复制游戏状态"""
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            move_count=self.move_count,
            game_over=self.game_over,
            winner=self.winner,
            move_history=self.move_history.copy()
        )

class SelfPlayEngine:
    """自我对弈引擎"""
    
    def __init__(self, neural_mcts_engine, config: SelfPlayConfig = None,
                 data_collector: TrainingDataCollector = None):
        """
        初始化自我对弈引擎
        
        Args:
            neural_mcts_engine: Neural MCTS引擎
            config: 自我对弈配置
            data_collector: 训练数据收集器
        """
        self.neural_mcts = neural_mcts_engine
        self.config = config or SelfPlayConfig()
        self.data_collector = data_collector or TrainingDataCollector()
        
        # 开局库
        self.opening_book = self._create_opening_book()
        
        # 统计信息
        self.stats = {
            'games_completed': 0,
            'total_moves': 0,
            'black_wins': 0,
            'white_wins': 0,
            'draws': 0,
            'avg_game_length': 0,
            'avg_time_per_move': 0,
            'total_time': 0
        }
        
        # 线程控制
        self.stop_flag = threading.Event()
        self.progress_callbacks: List[Callable] = []
        
        # 温度调度
        self.current_temperature = self.config.temperature
    
    def add_progress_callback(self, callback: Callable[[Dict], None]):
        """添加进度回调函数"""
        self.progress_callbacks.append(callback)
    
    def run_self_play(self, mode: SelfPlayMode = SelfPlayMode.SINGLE_THREAD) -> Dict:
        """
        运行自我对弈
        
        Args:
            mode: 对弈模式
            
        Returns:
            统计结果
        """
        print(f"开始自我对弈训练，模式: {mode.value}")
        print(f"目标游戏数: {self.config.num_games}")
        
        start_time = time.time()
        self.stop_flag.clear()
        
        try:
            if mode == SelfPlayMode.SINGLE_THREAD:
                self._run_single_thread()
            elif mode == SelfPlayMode.MULTI_THREAD:
                self._run_multi_thread()
            elif mode == SelfPlayMode.BATCH:
                self._run_batch_mode()
        
        except KeyboardInterrupt:
            print("\n自我对弈被用户中断")
            self.stop_flag.set()
        
        except Exception as e:
            print(f"自我对弈过程中出现错误: {e}")
            self.stop_flag.set()
        
        finally:
            total_time = time.time() - start_time
            self.stats['total_time'] = total_time
            
            # 计算平均值
            if self.stats['games_completed'] > 0:
                self.stats['avg_game_length'] = (
                    self.stats['total_moves'] / self.stats['games_completed']
                )
                self.stats['avg_time_per_move'] = (
                    total_time / max(1, self.stats['total_moves'])
                )
            
            print(f"\n自我对弈完成！")
            print(f"总时间: {total_time:.2f}秒")
            print(f"完成游戏数: {self.stats['games_completed']}")
            
            return self.stats
    
    def _run_single_thread(self):
        """单线程模式"""
        for game_idx in range(self.config.num_games):
            if self.stop_flag.is_set():
                break
            
            print(f"\n进行第 {game_idx + 1}/{self.config.num_games} 局游戏...")
            
            result = self._play_single_game(game_idx)
            self._update_stats(result)
            
            # 更新温度
            self._update_temperature()
            
            # 进度回调
            if self.config.enable_progress_callback:
                progress = {
                    'completed_games': game_idx + 1,
                    'total_games': self.config.num_games,
                    'progress_percent': ((game_idx + 1) / self.config.num_games) * 100,
                    'current_stats': self.stats.copy()
                }
                for callback in self.progress_callbacks:
                    try:
                        callback(progress)
                    except Exception as e:
                        print(f"进度回调错误: {e}")
            
            # 自动保存
            if (game_idx + 1) % self.config.save_frequency == 0:
                self._auto_save(game_idx + 1)
    
    def _run_multi_thread(self):
        """多线程模式"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            futures = []
            
            for game_idx in range(self.config.num_games):
                if self.stop_flag.is_set():
                    break
                
                future = executor.submit(self._play_single_game, game_idx)
                futures.append(future)
                
                # 控制并发数
                if len(futures) >= self.config.parallel_games:
                    # 等待部分任务完成
                    done_futures = []
                    for f in concurrent.futures.as_completed(futures[:self.config.parallel_games]):
                        result = f.result()
                        self._update_stats(result)
                        done_futures.append(f)
                    
                    # 移除已完成的任务
                    for f in done_futures:
                        futures.remove(f)
            
            # 等待剩余任务完成
            for future in concurrent.futures.as_completed(futures):
                if self.stop_flag.is_set():
                    break
                result = future.result()
                self._update_stats(result)
    
    def _run_batch_mode(self):
        """批处理模式"""
        batch_size = min(self.config.parallel_games, self.config.max_threads)
        
        for batch_start in range(0, self.config.num_games, batch_size):
            if self.stop_flag.is_set():
                break
            
            batch_end = min(batch_start + batch_size, self.config.num_games)
            batch_games = range(batch_start, batch_end)
            
            print(f"处理批次 {batch_start // batch_size + 1}: 游戏 {batch_start + 1}-{batch_end}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(self._play_single_game, game_idx) 
                          for game_idx in batch_games]
                
                for future in concurrent.futures.as_completed(futures):
                    if self.stop_flag.is_set():
                        break
                    result = future.result()
                    self._update_stats(result)
    
    def _play_single_game(self, game_idx: int) -> Dict:
        """
        进行单局游戏
        
        Args:
            game_idx: 游戏索引
            
        Returns:
            游戏结果
        """
        game_start_time = time.time()
        
        # 初始化游戏状态
        game_state = GameState(
            board=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int),
            current_player=BLACK
        )
        
        # 开始收集训练数据
        game_id = None
        if self.config.collect_training_data:
            game_id = self.data_collector.start_game(
                black_player=f"Neural_MCTS_SelfPlay_{game_idx}",
                white_player=f"Neural_MCTS_SelfPlay_{game_idx}",
                metadata={
                    'game_index': game_idx,
                    'temperature': self.current_temperature,
                    'mcts_simulations': self.config.mcts_simulations
                }
            )
        
        move_times = []
        
        # 游戏主循环
        for move_count in range(self.config.max_moves_per_game):
            if self.stop_flag.is_set():
                break
            
            move_start_time = time.time()
            
            # 获取可用移动
            available_moves = self._get_available_moves(game_state.board)
            if not available_moves:
                # 平局
                game_state.game_over = True
                game_state.winner = 0
                break
            
            # 选择移动
            if move_count < len(self.opening_book) and self.config.use_opening_book:
                # 使用开局库
                if self.config.randomize_opening:
                    move = random.choice(self.opening_book[move_count])
                else:
                    move = self.opening_book[move_count][0]
            else:
                # 使用MCTS
                move, move_info = self._get_mcts_move(game_state, available_moves)
                
                # 收集训练数据
                if self.config.collect_training_data and move_info:
                    self.data_collector.record_move(
                        position=move,
                        player=game_state.current_player,
                        board_state=game_state.board,
                        mcts_probabilities=move_info.get('probabilities', {}),
                        value_estimate=move_info.get('value_estimate', 0.0),
                        visit_counts=move_info.get('visit_counts', {}),
                        search_depth=move_info.get('search_depth', 0),
                        thinking_time=time.time() - move_start_time
                    )
            
            # 执行移动
            self._make_move(game_state, move)
            move_times.append(time.time() - move_start_time)
            
            # 检查胜负
            if self._check_winner(game_state.board, move):
                game_state.game_over = True
                game_state.winner = game_state.current_player
                break
            
            # 切换玩家
            game_state.current_player = 3 - game_state.current_player
            game_state.move_count += 1
        
        # 确定游戏结果
        if not game_state.game_over:
            # 超过最大步数，平局
            game_state.winner = 0
            game_state.game_over = True
        
        # 结束训练数据收集
        if self.config.collect_training_data and game_id:
            if game_state.winner == BLACK:
                result = GameResult.BLACK_WIN
            elif game_state.winner == WHITE:
                result = GameResult.WHITE_WIN
            else:
                result = GameResult.DRAW
            
            self.data_collector.end_game(result, time.time() - game_start_time)
        
        # 返回游戏结果
        return {
            'game_id': game_id,
            'winner': game_state.winner,
            'move_count': game_state.move_count,
            'game_time': time.time() - game_start_time,
            'avg_move_time': np.mean(move_times) if move_times else 0,
            'move_history': game_state.move_history
        }
    
    def _get_mcts_move(self, game_state: GameState, 
                      available_moves: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Dict]:
        """
        使用MCTS获取移动
        
        Args:
            game_state: 游戏状态
            available_moves: 可用移动
            
        Returns:
            (移动位置, 移动信息)
        """
        try:
            # 调用Neural MCTS引擎
            board_list = game_state.board.tolist()
            result = self.neural_mcts.get_best_move(board_list, game_state.current_player)
            
            if result and result.move:
                # 获取MCTS搜索信息
                move_info = {
                    'probabilities': getattr(result, 'move_probabilities', {}),
                    'value_estimate': getattr(result, 'value_estimate', 0.0),
                    'visit_counts': getattr(result, 'visit_counts', {}),
                    'search_depth': getattr(result, 'depth', 0)
                }
                
                # 应用温度采样
                if self.current_temperature > 0.1:
                    move = self._temperature_sampling(result, available_moves)
                else:
                    move = result.move
                
                return move, move_info
            
        except Exception as e:
            print(f"MCTS移动获取失败: {e}")
        
        # fallback：随机选择
        move = random.choice(available_moves)
        return move, {}
    
    def _temperature_sampling(self, mcts_result, available_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        温度采样选择移动
        
        Args:
            mcts_result: MCTS结果
            available_moves: 可用移动
            
        Returns:
            选择的移动
        """
        if not hasattr(mcts_result, 'move_probabilities') or not mcts_result.move_probabilities:
            return mcts_result.move if mcts_result.move else random.choice(available_moves)
        
        # 获取概率分布
        probabilities = mcts_result.move_probabilities
        
        # 应用温度
        if self.current_temperature <= 0.1:
            # 贪婪选择
            best_move = max(probabilities.items(), key=lambda x: x[1])[0]
            return best_move
        
        # 温度采样
        moves = []
        probs = []
        for move, prob in probabilities.items():
            if move in available_moves:
                moves.append(move)
                probs.append(prob ** (1.0 / self.current_temperature))
        
        if not moves:
            return random.choice(available_moves)
        
        # 归一化概率
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(probs)] * len(probs)
        
        # 采样
        return np.random.choice(moves, p=probs)
    
    def _get_available_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """获取可用移动"""
        moves = []
        
        # 如果是空棋盘，返回中心位置
        if np.all(board == EMPTY):
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        
        # 获取有邻居的空位
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY and self._has_neighbor(board, i, j):
                    moves.append((i, j))
        
        return moves
    
    def _has_neighbor(self, board: np.ndarray, row: int, col: int, radius: int = 2) -> bool:
        """检查是否有邻居"""
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = row + di, col + dj
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                    if board[ni][nj] != EMPTY:
                        return True
        return False
    
    def _make_move(self, game_state: GameState, move: Tuple[int, int]):
        """执行移动"""
        row, col = move
        game_state.board[row][col] = game_state.current_player
        game_state.move_history.append(move)
    
    def _check_winner(self, board: np.ndarray, last_move: Tuple[int, int]) -> bool:
        """检查是否有获胜者"""
        row, col = last_move
        player = board[row][col]
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
    
    def _create_opening_book(self) -> List[List[Tuple[int, int]]]:
        """创建开局库"""
        center = BOARD_SIZE // 2
        
        opening_book = [
            # 第一步：中心附近
            [(center, center), (center-1, center), (center+1, center), 
             (center, center-1), (center, center+1)],
            
            # 第二步：对角或邻近
            [(center-1, center-1), (center+1, center+1), (center-1, center+1), 
             (center+1, center-1), (center-2, center), (center+2, center)],
            
            # 第三步：形成攻击或防守
            [(center-2, center-1), (center+2, center+1), (center-1, center-2), 
             (center+1, center+2), (center, center-2), (center, center+2)]
        ]
        
        return opening_book
    
    def _update_temperature(self):
        """更新温度"""
        self.current_temperature = max(
            self.current_temperature * self.config.temperature_decay,
            self.config.min_temperature
        )
    
    def _update_stats(self, game_result: Dict):
        """更新统计信息"""
        self.stats['games_completed'] += 1
        self.stats['total_moves'] += game_result['move_count']
        
        if game_result['winner'] == BLACK:
            self.stats['black_wins'] += 1
        elif game_result['winner'] == WHITE:
            self.stats['white_wins'] += 1
        else:
            self.stats['draws'] += 1
    
    def _auto_save(self, completed_games: int):
        """自动保存"""
        try:
            filename = f"selfplay_data_{completed_games}_games.json"
            saved_file = self.data_collector.save_data(filename=filename, format_type="json")
            print(f"自动保存完成: {saved_file}")
        except Exception as e:
            print(f"自动保存失败: {e}")
    
    def stop_self_play(self):
        """停止自我对弈"""
        self.stop_flag.set()
        print("正在停止自我对弈...")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['current_temperature'] = self.current_temperature
        stats['data_collector_stats'] = self.data_collector.get_statistics()
        return stats
    
    def save_final_data(self) -> str:
        """保存最终数据"""
        filename = f"final_selfplay_data_{int(time.time())}.compressed"
        return self.data_collector.save_data(filename=filename, format_type="compressed")

def test_self_play_engine():
    """测试自我对弈引擎"""
    print("测试自我对弈引擎...")
    
    # 创建Mock Neural MCTS引擎
    class MockNeuralMCTS:
        def get_best_move(self, board, player):
            # 简单的随机移动模拟
            available_moves = []
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if board[i][j] == EMPTY:
                        available_moves.append((i, j))
            
            if available_moves:
                move = random.choice(available_moves)
                
                # 模拟结果对象
                class MockResult:
                    def __init__(self, move):
                        self.move = move
                        self.move_probabilities = {move: 0.8}
                        self.value_estimate = random.uniform(-0.5, 0.5)
                        self.visit_counts = {move: 800}
                        self.depth = 5
                
                return MockResult(move)
            return None
    
    # 创建配置
    config = SelfPlayConfig(
        num_games=3,  # 少量游戏用于测试
        max_moves_per_game=50,
        thinking_time_limit=0.1,
        mcts_simulations=100,
        save_frequency=2
    )
    
    # 创建引擎
    mock_mcts = MockNeuralMCTS()
    engine = SelfPlayEngine(mock_mcts, config)
    
    # 添加进度回调
    def progress_callback(progress):
        print(f"进度: {progress['completed_games']}/{progress['total_games']} "
              f"({progress['progress_percent']:.1f}%)")
    
    engine.add_progress_callback(progress_callback)
    
    # 运行自我对弈
    stats = engine.run_self_play(SelfPlayMode.SINGLE_THREAD)
    
    print(f"测试完成！统计信息: {stats}")
    
    # 保存最终数据
    final_file = engine.save_final_data()
    print(f"最终数据已保存: {final_file}")

if __name__ == "__main__":
    test_self_play_engine()