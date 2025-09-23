"""
神经网络评估函数模块
实现轻量级神经网络架构，可无缝集成到现有五子棋AI系统中

作者：Claude AI Engineer
日期：2025-09-22
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

# 游戏常量
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

class ActivationType(Enum):
    """激活函数类型"""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"

@dataclass
class NeuralConfig:
    """神经网络配置"""
    # 基础参数
    board_size: int = 15
    input_features: int = 32
    hidden_layers: List[int] = None
    output_size: int = 1
    
    # 激活函数
    activation: ActivationType = ActivationType.TANH
    
    # 特征配置
    use_pattern_features: bool = True
    use_position_features: bool = True
    use_threat_features: bool = True
    use_historical_features: bool = True
    
    # 训练参数
    learning_rate: float = 0.001
    dropout_rate: float = 0.1
    batch_norm: bool = True
    
    # 权重初始化
    weight_init_std: float = 0.1
    bias_init_value: float = 0.0
    
    def __post_init__(self):
        if self.hidden_layers is None:
            # 默认网络结构：输入 -> 128 -> 64 -> 32 -> 1
            self.hidden_layers = [128, 64, 32]

class NeuralLayer:
    """简单神经网络层实现"""
    
    def __init__(self, input_size: int, output_size: int, activation: ActivationType = ActivationType.TANH):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # 权重和偏置初始化
        self.weights = self._init_weights(input_size, output_size)
        self.biases = np.zeros(output_size)
        
        # 用于批量归一化的参数
        self.gamma = np.ones(output_size)
        self.beta = np.zeros(output_size)
        self.running_mean = np.zeros(output_size)
        self.running_var = np.ones(output_size)
        
        # 缓存中间结果用于反向传播
        self.last_input = None
        self.last_output = None
        
    def _init_weights(self, input_size: int, output_size: int) -> np.ndarray:
        """权重初始化（Xavier初始化）"""
        std = math.sqrt(2.0 / (input_size + output_size))
        return np.random.normal(0, std, (input_size, output_size))
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """激活函数"""
        if self.activation == ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == ActivationType.TANH:
            return np.tanh(x)
        elif self.activation == ActivationType.RELU:
            return np.maximum(0, x)
        elif self.activation == ActivationType.LEAKY_RELU:
            return np.where(x > 0, x, 0.01 * x)
        else:
            return x
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """前向传播"""
        self.last_input = x.copy()
        
        # 线性变换
        z = np.dot(x, self.weights) + self.biases
        
        # 批量归一化（简化版本）
        if training:
            mean = np.mean(z, axis=0)
            var = np.var(z, axis=0)
            # 更新运行统计
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
            self.running_var = (1 - momentum) * self.running_var + momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        z_norm = (z - mean) / np.sqrt(var + 1e-8)
        z_scaled = self.gamma * z_norm + self.beta
        
        # 激活
        output = self._activate(z_scaled)
        self.last_output = output.copy()
        
        return output

class FeatureExtractor:
    """五子棋特征提取器"""
    
    def __init__(self, config: NeuralConfig):
        self.config = config
        
        # 方向向量
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 棋型权重
        self.pattern_weights = {
            'five': 10000,
            'open_four': 5000,
            'blocked_four': 1000,
            'open_three': 500,
            'blocked_three': 100,
            'open_two': 50,
            'blocked_two': 10,
            'single': 1
        }
        
    def extract_features(self, board: List[List[int]], player: int, 
                        move_history: List[Tuple[int, int]] = None) -> np.ndarray:
        """提取棋盘特征"""
        features = []
        
        # 基础棋盘特征
        basic_features = self._extract_basic_features(board, player)
        features.extend(basic_features)
        
        # 棋型特征
        if self.config.use_pattern_features:
            pattern_features = self._extract_pattern_features(board, player)
            features.extend(pattern_features)
        
        # 位置特征
        if self.config.use_position_features:
            position_features = self._extract_position_features(board, player)
            features.extend(position_features)
        
        # 威胁特征
        if self.config.use_threat_features:
            threat_features = self._extract_threat_features(board, player)
            features.extend(threat_features)
        
        # 历史特征
        if self.config.use_historical_features and move_history:
            historical_features = self._extract_historical_features(board, move_history)
            features.extend(historical_features)
        
        # 确保特征向量长度一致
        target_length = self.config.input_features
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        elif len(features) > target_length:
            features = features[:target_length]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_basic_features(self, board: List[List[int]], player: int) -> List[float]:
        """提取基础特征"""
        features = []
        opponent = 3 - player
        
        # 棋子数量统计
        my_stones = sum(row.count(player) for row in board)
        opponent_stones = sum(row.count(opponent) for row in board)
        empty_spaces = sum(row.count(EMPTY) for row in board)
        
        total_spaces = BOARD_SIZE * BOARD_SIZE
        features.extend([
            my_stones / total_spaces,
            opponent_stones / total_spaces,
            empty_spaces / total_spaces,
            (my_stones - opponent_stones) / total_spaces
        ])
        
        # 中心控制
        center = BOARD_SIZE // 2
        center_control = 0
        for i in range(center - 2, center + 3):
            for j in range(center - 2, center + 3):
                if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                    if board[i][j] == player:
                        center_control += 1
                    elif board[i][j] == opponent:
                        center_control -= 1
        features.append(center_control / 25.0)  # 5x5中心区域
        
        return features
    
    def _extract_pattern_features(self, board: List[List[int]], player: int) -> List[float]:
        """提取棋型特征"""
        features = []
        opponent = 3 - player
        
        # 统计各种棋型
        my_patterns = self._count_patterns(board, player)
        opponent_patterns = self._count_patterns(board, opponent)
        
        # 添加特征
        for pattern_type in ['five', 'open_four', 'blocked_four', 'open_three', 
                           'blocked_three', 'open_two', 'blocked_two']:
            my_count = my_patterns.get(pattern_type, 0)
            opp_count = opponent_patterns.get(pattern_type, 0)
            
            # 归一化
            features.extend([
                my_count / 10.0,
                opp_count / 10.0,
                (my_count - opp_count) / 10.0
            ])
        
        return features
    
    def _count_patterns(self, board: List[List[int]], player: int) -> Dict[str, int]:
        """统计棋型数量"""
        patterns = {
            'five': 0, 'open_four': 0, 'blocked_four': 0,
            'open_three': 0, 'blocked_three': 0,
            'open_two': 0, 'blocked_two': 0
        }
        
        # 在所有方向上检查棋型
        for dx, dy in self.directions:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if board[i][j] == player:
                        line = self._get_line(board, i, j, dx, dy, 9)
                        pattern_type = self._analyze_line_pattern(line, player)
                        if pattern_type in patterns:
                            patterns[pattern_type] += 1
        
        return patterns
    
    def _get_line(self, board: List[List[int]], row: int, col: int, 
                  dx: int, dy: int, length: int) -> List[int]:
        """获取指定方向的线段"""
        line = []
        
        # 向两个方向延伸
        for direction in [-1, 1]:
            x, y = row, col
            for _ in range(length // 2):
                if direction == -1:
                    x -= dx
                    y -= dy
                else:
                    if not (x == row and y == col):  # 跳过起始点
                        x += dx
                        y += dy
                
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    if direction == -1:
                        line.insert(0, board[x][y])
                    else:
                        line.append(board[x][y])
                else:
                    # 边界
                    if direction == -1:
                        line.insert(0, -1)
                    else:
                        line.append(-1)
        
        # 添加中心点
        if len(line) == length - 1:
            line.insert(length // 2, board[row][col])
        
        return line
    
    def _analyze_line_pattern(self, line: List[int], player: int) -> str:
        """分析线段的棋型"""
        line_str = ''.join(str(x) for x in line)
        player_str = str(player)
        opponent_str = str(3 - player)
        
        # 检查各种棋型模式
        if player_str * 5 in line_str:
            return 'five'
        
        # 活四模式
        if f'0{player_str * 4}0' in line_str:
            return 'open_four'
        
        # 冲四模式
        if (f'{player_str * 4}0' in line_str or 
            f'0{player_str * 4}' in line_str or
            f'{player_str * 3}0{player_str}' in line_str or
            f'{player_str}0{player_str * 3}' in line_str):
            return 'blocked_four'
        
        # 活三模式
        if f'0{player_str * 3}0' in line_str:
            return 'open_three'
        
        # 眠三模式
        if (f'{player_str * 3}0' in line_str or 
            f'0{player_str * 3}' in line_str):
            return 'blocked_three'
        
        # 活二模式
        if f'0{player_str * 2}0' in line_str:
            return 'open_two'
        
        # 眠二模式
        if (f'{player_str * 2}0' in line_str or 
            f'0{player_str * 2}' in line_str):
            return 'blocked_two'
        
        return 'single'
    
    def _extract_position_features(self, board: List[List[int]], player: int) -> List[float]:
        """提取位置特征"""
        features = []
        
        # 重心位置
        my_positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == player:
                    my_positions.append((i, j))
        
        if my_positions:
            center_row = sum(pos[0] for pos in my_positions) / len(my_positions)
            center_col = sum(pos[1] for pos in my_positions) / len(my_positions)
            
            # 归一化到[-1, 1]
            features.extend([
                (center_row - BOARD_SIZE/2) / (BOARD_SIZE/2),
                (center_col - BOARD_SIZE/2) / (BOARD_SIZE/2)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # 分散度
        if len(my_positions) > 1:
            distances = []
            for i, pos1 in enumerate(my_positions):
                for pos2 in my_positions[i+1:]:
                    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    distances.append(dist)
            
            avg_distance = sum(distances) / len(distances)
            features.append(avg_distance / (BOARD_SIZE * math.sqrt(2)))
        else:
            features.append(0.0)
        
        return features
    
    def _extract_threat_features(self, board: List[List[int]], player: int) -> List[float]:
        """提取威胁特征"""
        features = []
        
        # 立即威胁计数
        immediate_threats = 0
        critical_threats = 0
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    # 模拟落子
                    board[i][j] = player
                    
                    # 检查是否能获胜
                    if self._check_win(board, i, j, player):
                        immediate_threats += 1
                    # 检查是否形成活四
                    elif self._check_open_four(board, i, j, player):
                        critical_threats += 1
                    
                    board[i][j] = EMPTY
        
        features.extend([
            immediate_threats / 10.0,
            critical_threats / 10.0
        ])
        
        return features
    
    def _check_win(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        for dx, dy in self.directions:
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
    
    def _check_open_four(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否形成活四"""
        for dx, dy in self.directions:
            line = self._get_line(board, row, col, dx, dy, 7)
            line_str = ''.join(str(x) for x in line)
            
            if f'0{str(player) * 4}0' in line_str:
                return True
        
        return False
    
    def _extract_historical_features(self, board: List[List[int]], 
                                   move_history: List[Tuple[int, int]]) -> List[float]:
        """提取历史特征"""
        features = []
        
        if not move_history:
            return [0.0] * 3
        
        # 最近移动的位置信息
        last_move = move_history[-1]
        center = BOARD_SIZE // 2
        
        # 最近移动距离中心的距离
        distance_to_center = math.sqrt(
            (last_move[0] - center)**2 + (last_move[1] - center)**2
        )
        features.append(distance_to_center / (BOARD_SIZE * math.sqrt(2)))
        
        # 移动趋势（是否向中心聚集）
        if len(move_history) >= 2:
            prev_move = move_history[-2]
            prev_distance = math.sqrt(
                (prev_move[0] - center)**2 + (prev_move[1] - center)**2
            )
            trend = (prev_distance - distance_to_center) / (BOARD_SIZE * math.sqrt(2))
            features.append(trend)
        else:
            features.append(0.0)
        
        # 移动密度（最近几步的平均距离）
        if len(move_history) >= 3:
            recent_moves = move_history[-3:]
            distances = []
            for i in range(len(recent_moves) - 1):
                dist = math.sqrt(
                    (recent_moves[i][0] - recent_moves[i+1][0])**2 +
                    (recent_moves[i][1] - recent_moves[i+1][1])**2
                )
                distances.append(dist)
            
            avg_distance = sum(distances) / len(distances) if distances else 0
            features.append(avg_distance / (BOARD_SIZE * math.sqrt(2)))
        else:
            features.append(0.0)
        
        return features

class NeuralNetworkEvaluator:
    """神经网络评估器"""
    
    def __init__(self, config: NeuralConfig = None):
        """初始化神经网络评估器"""
        self.config = config or NeuralConfig()
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(self.config)
        
        # 构建网络层
        self.layers = []
        self._build_network()
        
        # 性能统计
        self.evaluation_count = 0
        self.total_time = 0.0
        
        # 是否使用随机权重模拟训练
        self.use_random_weights = True
        
        print(f"神经网络评估器已初始化")
        print(f"网络结构: {self.config.input_features} -> {' -> '.join(map(str, self.config.hidden_layers))} -> {self.config.output_size}")
    
    def _build_network(self):
        """构建神经网络"""
        # 输入层到第一个隐藏层
        layer_sizes = [self.config.input_features] + self.config.hidden_layers + [self.config.output_size]
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # 输出层使用线性激活，其他层使用配置的激活函数
            activation = ActivationType.TANH if i < len(layer_sizes) - 2 else ActivationType.TANH
            
            layer = NeuralLayer(input_size, output_size, activation)
            self.layers.append(layer)
    
    def evaluate_board(self, board: List[List[int]], player: int) -> float:
        """评估棋盘状态"""
        start_time = time.time()
        
        # 提取特征
        features = self.feature_extractor.extract_features(board, player)
        
        # 前向传播
        output = self._forward(features)
        
        # 将输出映射到合适的范围
        score = float(output[0]) * 10000  # 放大到与传统评估函数类似的范围
        
        # 更新统计
        self.evaluation_count += 1
        self.total_time += time.time() - start_time
        
        return score
    
    def evaluate_move(self, board: List[List[int]], row: int, col: int, player: int) -> float:
        """评估特定移动"""
        if board[row][col] != EMPTY:
            return float('-inf')
        
        # 模拟落子
        board[row][col] = player
        score = self.evaluate_board(board, player)
        board[row][col] = EMPTY
        
        return score
    
    def _forward(self, features: np.ndarray) -> np.ndarray:
        """前向传播"""
        x = features.reshape(1, -1)  # 转换为批量维度
        
        for layer in self.layers:
            x = layer.forward(x, training=False)
        
        return x.flatten()
    
    def predict_move_probabilities(self, board: List[List[int]], player: int) -> Dict[Tuple[int, int], float]:
        """预测移动概率分布"""
        move_scores = {}
        total_score = 0
        
        # 计算所有合法移动的分数
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    score = self.evaluate_move(board, i, j, player)
                    # 使用softmax转换为概率
                    exp_score = math.exp(min(score / 1000, 700))  # 防止溢出
                    move_scores[(i, j)] = exp_score
                    total_score += exp_score
        
        # 归一化
        if total_score > 0:
            for move in move_scores:
                move_scores[move] /= total_score
        
        return move_scores
    
    def get_best_moves(self, board: List[List[int]], player: int, top_k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """获取最佳移动候选"""
        move_scores = []
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    score = self.evaluate_move(board, i, j, player)
                    move_scores.append(((i, j), score))
        
        # 按分数排序
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        return move_scores[:top_k]
    
    def update_weights_randomly(self, noise_scale: float = 0.1):
        """随机更新权重（模拟训练过程）"""
        for layer in self.layers:
            # 添加噪声到权重
            noise = np.random.normal(0, noise_scale, layer.weights.shape)
            layer.weights += noise
            
            # 添加噪声到偏置
            bias_noise = np.random.normal(0, noise_scale * 0.1, layer.biases.shape)
            layer.biases += bias_noise
    
    def save_weights(self, filepath: str):
        """保存网络权重"""
        weights_data = {
            'config': {
                'board_size': self.config.board_size,
                'input_features': self.config.input_features,
                'hidden_layers': self.config.hidden_layers,
                'output_size': self.config.output_size,
                'activation': self.config.activation.value
            },
            'layers': []
        }
        
        for i, layer in enumerate(self.layers):
            layer_data = {
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist(),
                'gamma': layer.gamma.tolist(),
                'beta': layer.beta.tolist(),
                'running_mean': layer.running_mean.tolist(),
                'running_var': layer.running_var.tolist()
            }
            weights_data['layers'].append(layer_data)
        
        with open(filepath, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        print(f"网络权重已保存到: {filepath}")
    
    def load_weights(self, filepath: str):
        """加载网络权重"""
        try:
            with open(filepath, 'r') as f:
                weights_data = json.load(f)
            
            # 验证配置兼容性
            saved_config = weights_data['config']
            if (saved_config['input_features'] != self.config.input_features or 
                saved_config['hidden_layers'] != self.config.hidden_layers):
                print("警告：保存的配置与当前配置不匹配，可能导致错误")
            
            # 加载层权重
            for i, layer_data in enumerate(weights_data['layers']):
                if i < len(self.layers):
                    layer = self.layers[i]
                    layer.weights = np.array(layer_data['weights'])
                    layer.biases = np.array(layer_data['biases'])
                    layer.gamma = np.array(layer_data['gamma'])
                    layer.beta = np.array(layer_data['beta'])
                    layer.running_mean = np.array(layer_data['running_mean'])
                    layer.running_var = np.array(layer_data['running_var'])
            
            print(f"网络权重已从 {filepath} 加载")
            
        except Exception as e:
            print(f"加载权重失败: {e}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        avg_time = self.total_time / max(1, self.evaluation_count)
        
        return {
            'evaluation_count': self.evaluation_count,
            'total_time': self.total_time,
            'avg_time_per_evaluation': avg_time,
            'evaluations_per_second': 1.0 / avg_time if avg_time > 0 else 0
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.evaluation_count = 0
        self.total_time = 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(
            layer.weights.size + layer.biases.size 
            for layer in self.layers
        )
        
        return {
            'model_type': 'NeuralNetworkEvaluator',
            'architecture': f"{self.config.input_features} -> {' -> '.join(map(str, self.config.hidden_layers))} -> {self.config.output_size}",
            'total_parameters': total_params,
            'activation_function': self.config.activation.value,
            'features_used': {
                'pattern_features': self.config.use_pattern_features,
                'position_features': self.config.use_position_features,
                'threat_features': self.config.use_threat_features,
                'historical_features': self.config.use_historical_features
            },
            'performance': self.get_performance_stats()
        }

# 与现有系统的集成接口
class NeuralEvaluatorAdapter:
    """神经网络评估器适配器，用于集成到现有系统"""
    
    def __init__(self, neural_evaluator: NeuralNetworkEvaluator, weight: float = 0.3):
        """
        初始化适配器
        
        Args:
            neural_evaluator: 神经网络评估器
            weight: 神经网络评估的权重（0-1之间）
        """
        self.neural_evaluator = neural_evaluator
        self.weight = weight
        self.backup_evaluator = None  # 可以设置备用评估器
        
    def evaluate_board(self, board: List[List[int]], player: int) -> int:
        """评估棋盘状态（兼容现有接口）"""
        try:
            neural_score = self.neural_evaluator.evaluate_board(board, player)
            return int(neural_score)
        except Exception as e:
            print(f"神经网络评估失败: {e}")
            # 回退到简单评估
            return self._fallback_evaluation(board, player)
    
    def evaluate_move(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估移动（兼容现有接口）"""
        try:
            neural_score = self.neural_evaluator.evaluate_move(board, row, col, player)
            return int(neural_score)
        except Exception as e:
            print(f"神经网络评估失败: {e}")
            return self._fallback_evaluation(board, player)
    
    def _fallback_evaluation(self, board: List[List[int]], player: int) -> int:
        """回退评估函数"""
        # 简单的棋子数量评估
        my_count = sum(row.count(player) for row in board)
        opponent_count = sum(row.count(3 - player) for row in board)
        return (my_count - opponent_count) * 100

# 测试和验证函数
def test_neural_evaluator():
    """测试神经网络评估器"""
    print("=== 测试神经网络评估器 ===")
    
    # 创建配置
    config = NeuralConfig(
        board_size=15,
        input_features=32,
        hidden_layers=[64, 32, 16],
        use_pattern_features=True,
        use_position_features=True,
        use_threat_features=True
    )
    
    # 创建评估器
    evaluator = NeuralNetworkEvaluator(config)
    
    # 创建测试棋盘
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # 设置一些棋子
    board[7][7] = BLACK
    board[7][8] = WHITE
    board[8][7] = WHITE
    board[8][8] = BLACK
    board[9][9] = BLACK
    
    print("测试棋盘评估...")
    score = evaluator.evaluate_board(board, BLACK)
    print(f"黑棋评估分数: {score:.2f}")
    
    print("测试移动评估...")
    move_score = evaluator.evaluate_move(board, 7, 9, BLACK)
    print(f"移动 (7,9) 评估分数: {move_score:.2f}")
    
    print("测试最佳移动推荐...")
    best_moves = evaluator.get_best_moves(board, BLACK, top_k=3)
    print("最佳移动候选:")
    for i, (move, score) in enumerate(best_moves):
        print(f"  {i+1}. {move}: {score:.2f}")
    
    print("测试移动概率预测...")
    probabilities = evaluator.predict_move_probabilities(board, BLACK)
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    print("移动概率前5:")
    for move, prob in sorted_probs:
        print(f"  {move}: {prob:.4f}")
    
    # 性能测试
    print("性能测试...")
    start_time = time.time()
    for _ in range(100):
        evaluator.evaluate_board(board, BLACK)
    end_time = time.time()
    
    print(f"100次评估用时: {end_time - start_time:.3f}秒")
    print(f"平均每次评估: {(end_time - start_time) / 100 * 1000:.2f}毫秒")
    
    # 获取模型信息
    model_info = evaluator.get_model_info()
    print("\n模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("\n=== 测试完成 ===")

def benchmark_against_traditional():
    """与传统评估函数进行基准测试"""
    print("=== 神经网络 vs 传统评估函数基准测试 ===")
    
    # 导入传统评估函数
    try:
        from advanced_evaluator import ComprehensiveEvaluator
        traditional_evaluator = ComprehensiveEvaluator()
    except ImportError:
        print("无法导入传统评估函数，跳过基准测试")
        return
    
    # 创建神经网络评估器
    neural_evaluator = NeuralNetworkEvaluator()
    
    # 创建测试场景
    test_scenarios = []
    
    # 场景1：开局
    board1 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board1[7][7] = BLACK
    test_scenarios.append(("开局", board1, BLACK))
    
    # 场景2：中局
    board2 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board2[7][7] = BLACK
    board2[7][8] = WHITE
    board2[8][7] = WHITE
    board2[8][8] = BLACK
    board2[6][6] = BLACK
    board2[9][9] = WHITE
    test_scenarios.append(("中局", board2, BLACK))
    
    # 场景3：残局
    board3 = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for i in range(5):
        board3[7][7+i] = BLACK if i % 2 == 0 else WHITE
        board3[8][7+i] = WHITE if i % 2 == 0 else BLACK
    test_scenarios.append(("残局", board3, BLACK))
    
    print(f"测试场景数: {len(test_scenarios)}")
    
    for scenario_name, board, player in test_scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # 神经网络评估
        start_time = time.time()
        neural_score = neural_evaluator.evaluate_board(board, player)
        neural_time = time.time() - start_time
        
        # 传统评估
        start_time = time.time()
        traditional_score = traditional_evaluator.comprehensive_evaluate(board, player)
        traditional_time = time.time() - start_time
        
        print(f"神经网络评估: {neural_score:.2f} (用时: {neural_time*1000:.2f}ms)")
        print(f"传统评估: {traditional_score:.2f} (用时: {traditional_time*1000:.2f}ms)")
        print(f"速度比: {traditional_time/neural_time:.2f}x")
        
        # 评估一致性
        correlation = 1.0 if (neural_score > 0) == (traditional_score > 0) else -1.0
        print(f"评估一致性: {'一致' if correlation > 0 else '不一致'}")

if __name__ == "__main__":
    # 运行测试
    test_neural_evaluator()
    
    print("\n" + "="*50 + "\n")
    
    # 运行基准测试
    benchmark_against_traditional()