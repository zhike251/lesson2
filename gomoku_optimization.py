"""
五子棋专用AI优化技术集成
结合传统算法和深度学习的混合方法

作者：Claude AI Engineer
日期：2025-09-22
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import math

# 导入已有模块
from alphazero_network import AlphaZeroNetwork, NetworkConfig
from neural_mcts import NeuralMCTS, AlphaZeroPlayer
from enhanced_resnet import AdvancedGomokuNetwork

class OptimizationTechnique(Enum):
    """优化技术枚举"""
    OPENING_BOOK = "opening_book"
    ENDGAME_TABLEBASE = "endgame_tablebase"
    THREAT_DETECTION = "threat_detection"
    PATTERN_MATCHING = "pattern_matching"
    POSITION_EVALUATION = "position_evaluation"
    SYMMETRY_BREAKING = "symmetry_breaking"
    MOVE_ORDERING = "move_ordering"
    EARLY_TERMINATION = "early_termination"

@dataclass
class OptimizationConfig:
    """优化配置"""
    use_opening_book: bool = True
    use_endgame_optimization: bool = True
    use_threat_detection: bool = True
    use_pattern_cache: bool = True
    use_symmetry_reduction: bool = True
    max_opening_moves: int = 8
    endgame_threshold: int = 20
    threat_search_depth: int = 4

class GomokuOpeningBook:
    """五子棋开局库"""
    
    def __init__(self):
        # 标准开局模式
        self.opening_patterns = {
            # 第一手：天元
            'center_start': {
                'sequence': [(7, 7)],
                'responses': [(6, 6), (6, 7), (6, 8), (7, 6), (7, 8), (8, 6), (8, 7), (8, 8)]
            },
            
            # 常见开局序列
            'standard_openings': [
                # 直指开局
                [(7, 7), (6, 7), (8, 7), (5, 7)],
                # 斜月开局  
                [(7, 7), (6, 6), (8, 8), (5, 5)],
                # 花月开局
                [(7, 7), (6, 7), (7, 6), (6, 6)],
                # 残月开局
                [(7, 7), (6, 7), (8, 6), (7, 6)]
            ]
        }
        
        # 开局评估分数
        self.opening_scores = {}
        self._initialize_opening_scores()
    
    def _initialize_opening_scores(self):
        """初始化开局评估分数"""
        # 基于专业五子棋理论的开局评分
        center = 7
        
        for i in range(15):
            for j in range(15):
                # 计算到中心的距离
                distance = max(abs(i - center), abs(j - center))
                
                # 距离越近分数越高
                if distance == 0:
                    score = 100  # 天元
                elif distance == 1:
                    score = 80   # 一线
                elif distance == 2:
                    score = 60   # 二线
                elif distance == 3:
                    score = 40   # 三线
                else:
                    score = 20   # 外围
                
                self.opening_scores[(i, j)] = score
    
    def get_opening_move(self, board: np.ndarray, move_count: int) -> Optional[Tuple[int, int]]:
        """
        获取开局移动建议
        
        Args:
            board: 当前棋盘
            move_count: 移动次数
            
        Returns:
            建议的移动位置
        """
        if move_count == 0:
            # 第一手：天元
            return (7, 7)
        
        if move_count < 8:
            # 在开局阶段，使用预定义的开局模式
            current_sequence = self._get_current_sequence(board)
            
            # 查找匹配的开局模式
            for opening in self.opening_patterns['standard_openings']:
                if self._matches_opening_prefix(current_sequence, opening):
                    next_move_index = len(current_sequence)
                    if next_move_index < len(opening):
                        candidate = opening[next_move_index]
                        if board[candidate[0], candidate[1]] == 0:
                            return candidate
            
            # 如果没有匹配的开局，选择评分最高的位置
            return self._get_best_opening_position(board)
        
        return None
    
    def _get_current_sequence(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """获取当前棋局序列"""
        sequence = []
        # 简化实现：按放置顺序记录（实际需要维护移动历史）
        for i in range(15):
            for j in range(15):
                if board[i, j] != 0:
                    sequence.append((i, j))
        return sequence
    
    def _matches_opening_prefix(self, current: List[Tuple[int, int]], 
                               pattern: List[Tuple[int, int]]) -> bool:
        """检查当前序列是否匹配开局模式前缀"""
        if len(current) > len(pattern):
            return False
        
        for i, move in enumerate(current):
            if i >= len(pattern) or move != pattern[i]:
                return False
        
        return True
    
    def _get_best_opening_position(self, board: np.ndarray) -> Tuple[int, int]:
        """获取开局阶段最佳位置"""
        best_score = -1
        best_move = None
        
        for i in range(15):
            for j in range(15):
                if board[i, j] == 0:
                    score = self.opening_scores.get((i, j), 0)
                    
                    # 考虑与已有棋子的距离
                    neighbor_bonus = self._calculate_neighbor_bonus(board, i, j)
                    total_score = score + neighbor_bonus
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_move = (i, j)
        
        return best_move
    
    def _calculate_neighbor_bonus(self, board: np.ndarray, row: int, col: int) -> int:
        """计算邻居奖励分数"""
        bonus = 0
        
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = row + di, col + dj
                if 0 <= ni < 15 and 0 <= nj < 15 and board[ni, nj] != 0:
                    distance = max(abs(di), abs(dj))
                    bonus += 20 // distance  # 距离越近奖励越高
        
        return bonus

class EndgameOptimizer:
    """终局优化器"""
    
    def __init__(self, tablebase_depth: int = 10):
        self.tablebase_depth = tablebase_depth
        self.tablebase = {}  # 简化的终局表
        
    def is_endgame(self, board: np.ndarray) -> bool:
        """判断是否进入终局"""
        empty_count = np.sum(board == 0)
        return empty_count <= 20  # 空位少于20个认为是终局
    
    def get_endgame_move(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """获取终局最佳移动"""
        if not self.is_endgame(board):
            return None
        
        # 简化的终局求解
        empty_positions = [(i, j) for i in range(15) for j in range(15) if board[i, j] == 0]
        
        if len(empty_positions) <= 5:
            # 暴力搜索所有可能
            return self._exhaustive_search(board, player, empty_positions)
        
        # 使用启发式方法
        return self._heuristic_endgame_search(board, player)
    
    def _exhaustive_search(self, board: np.ndarray, player: int, 
                          positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """穷尽搜索"""
        best_move = None
        best_score = float('-inf')
        
        for pos in positions:
            row, col = pos
            board[row, col] = player
            
            if self._check_win(board, row, col, player):
                board[row, col] = 0
                return pos
            
            score = self._minimax_endgame(board, 3 - player, len(positions) - 1, False)
            board[row, col] = 0
            
            if score > best_score:
                best_score = score
                best_move = pos
        
        return best_move
    
    def _heuristic_endgame_search(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """启发式终局搜索"""
        # 优先考虑能够获胜的移动
        for i in range(15):
            for j in range(15):
                if board[i, j] == 0:
                    board[i, j] = player
                    if self._check_win(board, i, j, player):
                        board[i, j] = 0
                        return (i, j)
                    board[i, j] = 0
        
        # 阻止对手获胜
        opponent = 3 - player
        for i in range(15):
            for j in range(15):
                if board[i, j] == 0:
                    board[i, j] = opponent
                    if self._check_win(board, i, j, opponent):
                        board[i, j] = 0
                        return (i, j)
                    board[i, j] = 0
        
        return None
    
    def _minimax_endgame(self, board: np.ndarray, player: int, depth: int, is_maximizing: bool) -> int:
        """终局极小极大搜索"""
        if depth == 0:
            return self._evaluate_endgame_position(board, player)
        
        empty_positions = [(i, j) for i in range(15) for j in range(15) if board[i, j] == 0]
        
        if is_maximizing:
            max_eval = float('-inf')
            for pos in empty_positions:
                row, col = pos
                board[row, col] = player
                
                if self._check_win(board, row, col, player):
                    board[row, col] = 0
                    return 1000
                
                eval_score = self._minimax_endgame(board, 3 - player, depth - 1, False)
                board[row, col] = 0
                max_eval = max(max_eval, eval_score)
            
            return max_eval
        else:
            min_eval = float('inf')
            for pos in empty_positions:
                row, col = pos
                board[row, col] = player
                
                if self._check_win(board, row, col, player):
                    board[row, col] = 0
                    return -1000
                
                eval_score = self._minimax_endgame(board, 3 - player, depth - 1, True)
                board[row, col] = 0
                min_eval = min(min_eval, eval_score)
            
            return min_eval
    
    def _evaluate_endgame_position(self, board: np.ndarray, player: int) -> int:
        """评估终局位置"""
        # 简化的终局评估
        score = 0
        
        # 计算连子威胁
        for i in range(15):
            for j in range(15):
                if board[i, j] == player:
                    score += self._count_threats_at_position(board, i, j, player)
                elif board[i, j] == 3 - player:
                    score -= self._count_threats_at_position(board, i, j, 3 - player)
        
        return score
    
    def _count_threats_at_position(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """计算位置的威胁数量"""
        threats = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < 15 and 0 <= y < 15 and board[x, y] == player:
                count += 1
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while 0 <= x < 15 and 0 <= y < 15 and board[x, y] == player:
                count += 1
                x -= dx
                y -= dy
            
            if count >= 4:
                threats += 10
            elif count >= 3:
                threats += 3
            elif count >= 2:
                threats += 1
        
        return threats
    
    def _check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < 15 and 0 <= y < 15 and board[x, y] == player:
                count += 1
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while 0 <= x < 15 and 0 <= y < 15 and board[x, y] == player:
                count += 1
                x -= dx
                y -= dy
            
            if count >= 5:
                return True
        
        return False

class HybridGomokuAI:
    """混合五子棋AI系统"""
    
    def __init__(self, neural_network, optimization_config: OptimizationConfig):
        self.neural_network = neural_network
        self.config = optimization_config
        
        # 初始化各个组件
        self.opening_book = GomokuOpeningBook() if optimization_config.use_opening_book else None
        self.endgame_optimizer = EndgameOptimizer() if optimization_config.use_endgame_optimization else None
        
        # MCTS组件
        self.mcts = NeuralMCTS(neural_network)
        
        # 统计信息
        self.stats = {
            'opening_book_hits': 0,
            'endgame_solver_hits': 0,
            'neural_network_calls': 0,
            'total_moves': 0
        }
    
    def get_best_move(self, board: np.ndarray, player: int) -> Tuple[Tuple[int, int], Dict]:
        """
        获取最佳移动（混合策略）
        
        Args:
            board: 当前棋盘状态
            player: 当前玩家
            
        Returns:
            (move, info): 最佳移动和信息
        """
        start_time = time.time()
        move_info = {'method': 'unknown', 'confidence': 0.0, 'search_time': 0.0}
        
        # 统计移动次数
        move_count = np.sum(board != 0)
        self.stats['total_moves'] += 1
        
        # 1. 开局阶段：使用开局库
        if (self.opening_book and 
            move_count <= self.config.max_opening_moves):
            
            opening_move = self.opening_book.get_opening_move(board, move_count)
            if opening_move and board[opening_move[0], opening_move[1]] == 0:
                move_info.update({
                    'method': 'opening_book',
                    'confidence': 0.9,
                    'search_time': time.time() - start_time
                })
                self.stats['opening_book_hits'] += 1
                return opening_move, move_info
        
        # 2. 终局阶段：使用终局求解器
        if (self.endgame_optimizer and 
            self.endgame_optimizer.is_endgame(board)):
            
            endgame_move = self.endgame_optimizer.get_endgame_move(board, player)
            if endgame_move:
                move_info.update({
                    'method': 'endgame_solver',
                    'confidence': 0.95,
                    'search_time': time.time() - start_time
                })
                self.stats['endgame_solver_hits'] += 1
                return endgame_move, move_info
        
        # 3. 中局阶段：使用神经网络+MCTS
        action_probs, root_value = self.mcts.search(board, player)
        
        # 选择最佳移动
        if action_probs:
            best_move = max(action_probs.keys(), key=lambda a: action_probs[a])
            confidence = action_probs[best_move]
        else:
            # 备用策略：选择中心附近的空位
            best_move = self._fallback_move(board)
            confidence = 0.1
        
        move_info.update({
            'method': 'neural_mcts',
            'confidence': confidence,
            'search_time': time.time() - start_time,
            'root_value': root_value,
            'action_probs': action_probs
        })
        
        self.stats['neural_network_calls'] += 1
        return best_move, move_info
    
    def _fallback_move(self, board: np.ndarray) -> Tuple[int, int]:
        """备用移动策略"""
        # 选择中心附近的空位
        center = 7
        for distance in range(8):
            for i in range(max(0, center - distance), min(15, center + distance + 1)):
                for j in range(max(0, center - distance), min(15, center + distance + 1)):
                    if board[i, j] == 0:
                        return (i, j)
        
        # 如果都没有，随机选择
        empty_positions = [(i, j) for i in range(15) for j in range(15) if board[i, j] == 0]
        if empty_positions:
            return empty_positions[0]
        
        return (0, 0)  # 应该不会到达这里
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_moves = max(1, self.stats['total_moves'])
        
        return {
            'total_moves': total_moves,
            'opening_book_usage': self.stats['opening_book_hits'] / total_moves,
            'endgame_solver_usage': self.stats['endgame_solver_hits'] / total_moves,
            'neural_network_usage': self.stats['neural_network_calls'] / total_moves,
            'mcts_statistics': self.mcts.get_statistics() if hasattr(self.mcts, 'get_statistics') else {}
        }

class AdaptiveTrainingSystem:
    """自适应训练系统"""
    
    def __init__(self, base_network: nn.Module):
        self.base_network = base_network
        self.training_history = []
        self.performance_metrics = {}
        
    def adaptive_training_loop(self, num_iterations: int):
        """自适应训练循环"""
        for iteration in range(num_iterations):
            # 动态调整训练参数
            learning_rate = self._adaptive_learning_rate(iteration)
            temperature = self._adaptive_temperature(iteration)
            mcts_simulations = self._adaptive_mcts_simulations(iteration)
            
            # 执行训练步骤
            training_loss = self._training_step(learning_rate, temperature, mcts_simulations)
            
            # 记录性能
            self.training_history.append({
                'iteration': iteration,
                'loss': training_loss,
                'learning_rate': learning_rate,
                'temperature': temperature,
                'mcts_simulations': mcts_simulations
            })
            
            # 评估和调整
            if iteration % 10 == 0:
                self._evaluate_and_adjust(iteration)
    
    def _adaptive_learning_rate(self, iteration: int) -> float:
        """自适应学习率"""
        base_lr = 0.001
        decay_rate = 0.95
        decay_steps = 50
        
        return base_lr * (decay_rate ** (iteration // decay_steps))
    
    def _adaptive_temperature(self, iteration: int) -> float:
        """自适应温度参数"""
        if iteration < 50:
            return 1.0  # 初期高温度，增加探索
        elif iteration < 200:
            return 0.5  # 中期中等温度
        else:
            return 0.1  # 后期低温度，增加利用
    
    def _adaptive_mcts_simulations(self, iteration: int) -> int:
        """自适应MCTS模拟次数"""
        if iteration < 100:
            return 400   # 初期较少模拟
        elif iteration < 500:
            return 800   # 中期标准模拟
        else:
            return 1600  # 后期增加模拟精度
    
    def _training_step(self, learning_rate: float, temperature: float, mcts_simulations: int) -> float:
        """执行一步训练"""
        # 这里应该实现具体的训练逻辑
        # 简化版本返回模拟损失
        return np.random.exponential(1.0)  # 模拟指数分布的损失
    
    def _evaluate_and_adjust(self, iteration: int):
        """评估并调整训练策略"""
        recent_losses = [entry['loss'] for entry in self.training_history[-10:]]
        avg_loss = np.mean(recent_losses)
        
        # 如果损失不再下降，调整策略
        if len(self.training_history) > 20:
            older_losses = [entry['loss'] for entry in self.training_history[-20:-10]]
            older_avg = np.mean(older_losses)
            
            if avg_loss >= older_avg * 0.95:  # 损失下降缓慢
                print(f"迭代 {iteration}: 检测到收敛减慢，考虑调整策略")

# 测试和演示代码
def test_hybrid_ai():
    """测试混合AI系统"""
    print("测试混合五子棋AI系统...")
    
    # 创建网络配置
    config = NetworkConfig(board_size=15, input_channels=8)
    network = AlphaZeroNetwork(config)
    
    # 创建优化配置
    opt_config = OptimizationConfig(
        use_opening_book=True,
        use_endgame_optimization=True,
        use_threat_detection=True
    )
    
    # 创建混合AI
    hybrid_ai = HybridGomokuAI(network, opt_config)
    
    # 测试不同阶段
    print("\n1. 测试开局阶段:")
    empty_board = np.zeros((15, 15))
    move, info = hybrid_ai.get_best_move(empty_board, 1)
    print(f"开局移动: {move}, 方法: {info['method']}, 置信度: {info['confidence']:.3f}")
    
    print("\n2. 测试中局阶段:")
    mid_board = np.zeros((15, 15))
    mid_board[7, 7] = 1
    mid_board[7, 8] = 2
    mid_board[8, 7] = 1
    mid_board[8, 8] = 2
    move, info = hybrid_ai.get_best_move(mid_board, 1)
    print(f"中局移动: {move}, 方法: {info['method']}, 置信度: {info['confidence']:.3f}")
    
    print("\n3. 测试终局阶段:")
    end_board = np.random.choice([0, 1, 2], size=(15, 15), p=[0.1, 0.45, 0.45])
    move, info = hybrid_ai.get_best_move(end_board, 1)
    print(f"终局移动: {move}, 方法: {info['method']}, 置信度: {info['confidence']:.3f}")
    
    # 获取统计信息
    stats = hybrid_ai.get_statistics()
    print(f"\n4. AI统计信息:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_hybrid_ai()