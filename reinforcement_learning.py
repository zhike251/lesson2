#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自博弈强化学习系统
基于AlphaZero算法的自博弈训练框架

作者：Claude AI Engineer
日期：2025-09-23
"""

import numpy as np
import time
import json
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import os
import pickle

from neural_mcts import NeuralMCTS, AlphaZeroStyleNetwork
from neural_evaluator import NeuralNetworkEvaluator, NeuralConfig
from training_data_collector import TrainingDataCollector, GameRecord

@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    num_iterations: int = 1000  # 训练迭代次数
    games_per_iteration: int = 100  # 每次迭代的自博弈游戏数
    batch_size: int = 64  # 训练批次大小
    epochs_per_iteration: int = 10  # 每次迭代训练轮数
    
    # MCTS参数
    mcts_simulations: int = 800
    c_puct: float = 1.25
    temperature: float = 1.0
    temperature_drop: float = 0.5
    
    # 神经网络参数
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout_rate: float = 0.3
    
    # 经验回放
    replay_buffer_size: int = 10000
    replay_buffer_sample_size: int = 1000
    
    # 保存和评估
    save_interval: int = 10  # 每10次迭代保存一次
    evaluation_interval: int = 5  # 每5次迭代评估一次
    evaluation_games: int = 50  # 评估游戏数
    
    # 路径配置
    model_save_path: str = "reinforcement_models"
    data_save_path: str = "reinforcement_data"
    log_path: str = "reinforcement_logs"

@dataclass
class GameExperience:
    """游戏经验"""
    board_state: np.ndarray
    policy_target: np.ndarray
    value_target: float
    game_result: float
    move_number: int
    
class SelfPlayEngine:
    """自博弈引擎"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.board_size = 15
        
        # 初始化神经网络
        self.neural_network = self._create_neural_network()
        
        # 初始化MCTS
        self.mcts = NeuralMCTS(
            network=self.neural_network,
            num_simulations=config.mcts_simulations,
            c_puct=config.c_puct
        )
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # 训练统计
        self.training_stats = {
            'iteration': 0,
            'total_games': 0,
            'total_moves': 0,
            'win_rates': [],
            'loss_history': [],
            'evaluation_results': []
        }
        
        # 创建保存目录
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.data_save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        
    def _create_neural_network(self) -> AlphaZeroStyleNetwork:
        """创建神经网络"""
        # 创建高级神经网络配置
        neural_config = NeuralConfig(
            board_size=15,
            input_features=48,  # 增加输入特征
            hidden_layers=[256, 128, 64],  # 更深的网络
            use_pattern_features=True,
            use_position_features=True,
            use_threat_features=True,
            use_historical_features=True,
            use_global_features=True  # 添加全局特征
        )
        
        # 创建神经网络评估器
        neural_evaluator = NeuralNetworkEvaluator(neural_config)
        
        # 创建AlphaZero风格网络适配器
        network = AlphaZeroStyleNetwork(neural_evaluator)
        
        return network
    
    def play_self_play_game(self, temperature: float = 1.0) -> List[GameExperience]:
        """进行一局自博弈游戏"""
        board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        current_player = 1  # 1为黑棋，2为白棋
        
        game_experiences = []
        move_history = []
        
        while not self._is_game_over(board):
            # MCTS搜索
            root = self.mcts.create_root(board, current_player)
            self.mcts.run_simulation(root)
            
            # 获取移动概率
            move_probabilities = self.mcts.get_move_probabilities(root, temperature)
            
            # 选择移动
            move = self._select_move(move_probabilities)
            
            # 记录经验
            if move:
                board_state = self._encode_board_state(board)
                policy_target = self._create_policy_target(move_probabilities)
                
                experience = GameExperience(
                    board_state=board_state,
                    policy_target=policy_target,
                    value_target=0.0,  # 游戏结束时再设置
                    game_result=0.0,   # 游戏结束时再设置
                    move_number=len(move_history)
                )
                game_experiences.append(experience)
                
                # 执行移动
                row, col = move
                board[row][col] = current_player
                move_history.append(move)
                
                # 切换玩家
                current_player = 3 - current_player
            else:
                break
        
        # 游戏结束，设置最终结果
        winner = self._determine_winner(board)
        game_result = 1.0 if winner == 1 else -1.0 if winner == 2 else 0.0
        
        # 更新所有经验的价值目标
        for experience in game_experiences:
            experience.game_result = game_result
            # 从最后一步向前传播价值
            if winner == 1:
                experience.value_target = 1.0 if experience.move_number % 2 == 0 else -1.0
            elif winner == 2:
                experience.value_target = -1.0 if experience.move_number % 2 == 0 else 1.0
            else:
                experience.value_target = 0.0
        
        return game_experiences
    
    def _encode_board_state(self, board: List[List[int]]) -> np.ndarray:
        """编码棋盘状态"""
        # 创建多通道输入
        channels = 3  # 当前玩家棋子、对手棋子、空位
        state = np.zeros((channels, self.board_size, self.board_size))
        
        # 当前玩家视角
        state[0] = np.array([[1 if cell == 1 else 0 for cell in row] for row in board])
        state[1] = np.array([[1 if cell == 2 else 0 for cell in row] for row in board])
        state[2] = np.array([[1 if cell == 0 else 0 for cell in row] for row in board])
        
        return state
    
    def _create_policy_target(self, move_probabilities: Dict[Tuple[int, int], float]) -> np.ndarray:
        """创建策略目标"""
        policy = np.zeros((self.board_size, self.board_size))
        
        for (row, col), prob in move_probabilities.items():
            policy[row][col] = prob
        
        # 归一化
        if np.sum(policy) > 0:
            policy = policy / np.sum(policy)
        
        return policy
    
    def _select_move(self, move_probabilities: Dict[Tuple[int, int], float]) -> Optional[Tuple[int, int]]:
        """根据概率选择移动"""
        if not move_probabilities:
            return None
        
        moves = list(move_probabilities.keys())
        probabilities = list(move_probabilities.values())
        
        # 添加一些随机性以避免总是选择最佳移动
        if random.random() < 0.1:  # 10%的概率随机选择
            return random.choice(moves)
        
        # 根据概率选择
        return random.choices(moves, weights=probabilities)[0]
    
    def _is_game_over(self, board: List[List[int]]) -> bool:
        """检查游戏是否结束"""
        # 检查是否有玩家获胜
        if self._check_winner(board, 1) or self._check_winner(board, 2):
            return True
        
        # 检查是否平局
        for row in board:
            for cell in row:
                if cell == 0:
                    return False
        
        return True
    
    def _check_winner(self, board: List[List[int]], player: int) -> bool:
        """检查指定玩家是否获胜"""
        # 检查所有方向的五连
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == player:
                    for dx, dy in directions:
                        if self._check_line(board, row, col, dx, dy, player):
                            return True
        
        return False
    
    def _check_line(self, board: List[List[int]], row: int, col: int, 
                    dx: int, dy: int, player: int) -> bool:
        """检查某个方向是否有五连"""
        count = 1
        
        # 正向检查
        x, y = row + dx, col + dy
        while (0 <= x < self.board_size and 0 <= y < self.board_size and 
               board[x][y] == player):
            count += 1
            x += dx
            y += dy
        
        # 反向检查
        x, y = row - dx, col - dy
        while (0 <= x < self.board_size and 0 <= y < self.board_size and 
               board[x][y] == player):
            count += 1
            x -= dx
            y -= dy
        
        return count >= 5
    
    def _determine_winner(self, board: List[List[int]]) -> int:
        """确定获胜者"""
        if self._check_winner(board, 1):
            return 1
        elif self._check_winner(board, 2):
            return 2
        else:
            return 0  # 平局
    
    def add_experience(self, experiences: List[GameExperience]):
        """添加经验到回放缓冲区"""
        for experience in experiences:
            self.replay_buffer.append(experience)
    
    def sample_experience_batch(self, batch_size: int) -> List[GameExperience]:
        """从回放缓冲区采样经验批次"""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        
        return random.sample(list(self.replay_buffer), batch_size)
    
    def train_network(self, experiences: List[GameExperience]) -> Dict[str, float]:
        """训练神经网络"""
        if not experiences:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        
        # 准备训练数据
        board_states = np.array([exp.board_state for exp in experiences])
        policy_targets = np.array([exp.policy_target.flatten() for exp in experiences])
        value_targets = np.array([exp.value_target for exp in experiences])
        
        # 训练神经网络
        loss_info = self.neural_network.train_batch(
            board_states, policy_targets, value_targets
        )
        
        return loss_info
    
    def evaluate_network(self, num_games: int = 50) -> float:
        """评估网络性能"""
        wins = 0
        losses = 0
        draws = 0
        
        for _ in range(num_games):
            # 与简单对手对战
            result = self.play_evaluation_game()
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        
        win_rate = wins / num_games
        return win_rate
    
    def play_evaluation_game(self) -> int:
        """进行一局评估游戏"""
        # 简化的评估游戏逻辑
        board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        current_player = 1
        
        for _ in range(50):  # 限制步数
            # 简单的随机移动策略
            empty_positions = [(i, j) for i in range(self.board_size) 
                             for j in range(self.board_size) if board[i][j] == 0]
            
            if not empty_positions:
                break
            
            if current_player == 1:
                # 使用训练的网络
                move = self._get_network_move(board)
            else:
                # 使用随机移动
                move = random.choice(empty_positions)
            
            if move:
                row, col = move
                board[row][col] = current_player
                
                if self._check_winner(board, current_player):
                    return 1 if current_player == 1 else -1
                
                current_player = 3 - current_player
        
        return 0  # 平局
    
    def _get_network_move(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """使用网络获取移动"""
        try:
            board_state = self._encode_board_state(board)
            policy, value = self.neural_network.predict(board_state)
            
            # 选择概率最高的有效移动
            policy_2d = policy.reshape(self.board_size, self.board_size)
            
            best_move = None
            best_prob = -1
            
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == 0 and policy_2d[i][j] > best_prob:
                        best_prob = policy_2d[i][j]
                        best_move = (i, j)
            
            return best_move
        except:
            # 如果网络预测失败，返回中心位置
            return (self.board_size // 2, self.board_size // 2)
    
    def save_model(self, iteration: int):
        """保存模型"""
        model_path = os.path.join(self.config.model_save_path, f"model_{iteration:04d}.pkl")
        
        model_data = {
            'iteration': iteration,
            'network_state': self.neural_network.get_state(),
            'training_stats': self.training_stats,
            'config': asdict(self.config)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.neural_network.set_state(model_data['network_state'])
        self.training_stats = model_data['training_stats']
        
        print(f"模型已从 {model_path} 加载")
    
    def save_training_data(self, iteration: int):
        """保存训练数据"""
        data_path = os.path.join(self.config.data_save_path, f"training_data_{iteration:04d}.json")
        
        # 转换经验数据为可序列化格式
        experience_data = []
        for exp in self.replay_buffer:
            experience_data.append({
                'board_state': exp.board_state.tolist(),
                'policy_target': exp.policy_target.tolist(),
                'value_target': exp.value_target,
                'game_result': exp.game_result,
                'move_number': exp.move_number
            })
        
        data = {
            'iteration': iteration,
            'experiences': experience_data[-1000:],  # 只保存最近1000个经验
            'training_stats': self.training_stats
        }
        
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"训练数据已保存到: {data_path}")

class ReinforcementLearningTrainer:
    """强化学习训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.engine = SelfPlayEngine(config)
        
    def train(self):
        """开始训练"""
        print(f"开始强化学习训练，共 {self.config.num_iterations} 次迭代")
        
        for iteration in range(self.config.num_iterations):
            print(f"\n=== 第 {iteration + 1} 次迭代 ===")
            
            # 自博弈阶段
            print(f"进行 {self.config.games_per_iteration} 局自博弈游戏...")
            all_experiences = []
            
            for game in range(self.config.games_per_iteration):
                if game % 10 == 0:
                    print(f"  游戏进度: {game}/{self.config.games_per_iteration}")
                
                # 动态调整温度参数
                temperature = max(self.config.temperature_drop, 
                                self.config.temperature * (0.99 ** iteration))
                
                experiences = self.engine.play_self_play_game(temperature)
                all_experiences.extend(experiences)
                self.engine.add_experience(experiences)
            
            print(f"收集到 {len(all_experiences)} 个经验")
            
            # 训练阶段
            print(f"训练神经网络，批次大小: {self.config.batch_size}")
            
            for epoch in range(self.config.epochs_per_iteration):
                # 从经验回放缓冲区采样
                batch_experiences = self.engine.sample_experience_batch(self.config.batch_size)
                
                # 训练网络
                loss_info = self.engine.train_network(batch_experiences)
                
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch + 1}/{self.config.epochs_per_iteration}: "
                          f"Loss={loss_info['loss']:.4f}, "
                          f"Policy={loss_info['policy_loss']:.4f}, "
                          f"Value={loss_info['value_loss']:.4f}")
            
            # 更新统计信息
            self.engine.training_stats['iteration'] = iteration + 1
            self.engine.training_stats['total_games'] += self.config.games_per_iteration
            self.engine.training_stats['total_moves'] += len(all_experiences)
            
            # 评估阶段
            if (iteration + 1) % self.config.evaluation_interval == 0:
                print("评估网络性能...")
                win_rate = self.engine.evaluate_network(self.config.evaluation_games)
                self.engine.training_stats['evaluation_results'].append(win_rate)
                print(f"胜率: {win_rate:.2%}")
            
            # 保存阶段
            if (iteration + 1) % self.config.save_interval == 0:
                self.engine.save_model(iteration + 1)
                self.engine.save_training_data(iteration + 1)
            
            print(f"第 {iteration + 1} 次迭代完成")
        
        print("训练完成！")
        
        # 保存最终模型
        self.engine.save_model('final')
        self.engine.save_training_data('final')
        
        return self.engine

if __name__ == "__main__":
    # 创建训练配置
    config = TrainingConfig(
        num_iterations=100,  # 快速测试
        games_per_iteration=10,
        batch_size=32,
        epochs_per_iteration=5,
        save_interval=5,
        evaluation_interval=2
    )
    
    # 开始训练
    trainer = ReinforcementLearningTrainer(config)
    trained_engine = trainer.train()
    
    print("强化学习训练完成！")