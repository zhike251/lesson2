"""
AlphaZero风格的神经网络架构
包含策略网络和价值网络的残差网络实现

作者：Claude AI Engineer
日期：2025-09-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """网络配置"""
    board_size: int = 15
    input_channels: int = 8  # 历史层数
    residual_blocks: int = 10
    filters: int = 256
    policy_head_filters: int = 2
    value_head_filters: int = 1
    dropout_rate: float = 0.3

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, filters: int, dropout_rate: float = 0.0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        if self.dropout:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # 残差连接
        out = F.relu(out)
        
        return out

class PolicyHead(nn.Module):
    """策略头部"""
    
    def __init__(self, config: NetworkConfig):
        super(PolicyHead, self).__init__()
        
        self.config = config
        
        # 卷积层
        self.conv = nn.Conv2d(
            config.filters, 
            config.policy_head_filters, 
            kernel_size=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(config.policy_head_filters)
        
        # 全连接层
        conv_output_size = config.policy_head_filters * config.board_size * config.board_size
        self.fc = nn.Linear(conv_output_size, config.board_size * config.board_size)
        
    def forward(self, x):
        # 卷积处理
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接
        out = self.fc(out)
        
        # 应用softmax得到概率分布
        policy = F.softmax(out, dim=1)
        
        return policy

class ValueHead(nn.Module):
    """价值头部"""
    
    def __init__(self, config: NetworkConfig):
        super(ValueHead, self).__init__()
        
        self.config = config
        
        # 卷积层
        self.conv = nn.Conv2d(
            config.filters, 
            config.value_head_filters, 
            kernel_size=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(config.value_head_filters)
        
        # 全连接层
        conv_output_size = config.value_head_filters * config.board_size * config.board_size
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x):
        # 卷积处理
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        # 应用tanh得到-1到1的价值
        value = torch.tanh(out)
        
        return value

class AlphaZeroNetwork(nn.Module):
    """AlphaZero风格的神经网络"""
    
    def __init__(self, config: NetworkConfig):
        super(AlphaZeroNetwork, self).__init__()
        
        self.config = config
        
        # 输入卷积层
        self.input_conv = nn.Conv2d(
            config.input_channels, 
            config.filters, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(config.filters)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.filters, config.dropout_rate) 
            for _ in range(config.residual_blocks)
        ])
        
        # 策略和价值头部
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        
    def forward(self, x):
        # 输入处理
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)
        
        # 通过残差块
        for block in self.residual_blocks:
            out = block(out)
        
        # 策略和价值预测
        policy = self.policy_head(out)
        value = self.value_head(out)
        
        return policy, value
    
    def predict(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        预测策略概率和价值（兼容MCTS接口）
        
        Args:
            state: 棋盘状态 (15, 15)
            player: 当前玩家
            
        Returns:
            (policy_probs, value): 策略概率字典和价值估计
        """
        self.eval()
        
        # 准备输入
        input_tensor = self._prepare_input(state, player)
        
        with torch.no_grad():
            policy_logits, value = self.forward(input_tensor)
        
        # 转换为字典格式
        policy_probs = self._convert_policy_to_dict(policy_logits[0], state)
        value_scalar = value.item()
        
        return policy_probs, value_scalar
    
    def _prepare_input(self, state: np.ndarray, player: int) -> torch.Tensor:
        """
        准备网络输入
        
        Args:
            state: 棋盘状态
            player: 当前玩家
            
        Returns:
            网络输入张量 (1, channels, height, width)
        """
        batch_size = 1
        channels = self.config.input_channels
        height, width = self.config.board_size, self.config.board_size
        
        # 创建输入张量
        input_tensor = torch.zeros(batch_size, channels, height, width)
        
        # 通道0：当前玩家的棋子
        input_tensor[0, 0] = torch.from_numpy((state == player).astype(np.float32))
        
        # 通道1：对手的棋子
        input_tensor[0, 1] = torch.from_numpy((state == (3 - player)).astype(np.float32))
        
        # 通道2：空位
        input_tensor[0, 2] = torch.from_numpy((state == 0).astype(np.float32))
        
        # 通道3：当前玩家标识（全1或全0）
        input_tensor[0, 3] = float(player == 1)
        
        # 通道4-7：历史信息（简化版本）
        for i in range(4, min(8, channels)):
            input_tensor[0, i] = input_tensor[0, 0]  # 复制当前玩家信息
        
        return input_tensor
    
    def _convert_policy_to_dict(self, policy_logits: torch.Tensor, state: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        将策略logits转换为位置概率字典
        
        Args:
            policy_logits: 策略概率 (board_size * board_size,)
            state: 棋盘状态
            
        Returns:
            位置概率字典
        """
        policy_dict = {}
        
        # 重塑为棋盘形状
        policy_2d = policy_logits.view(self.config.board_size, self.config.board_size)
        
        # 只保留合法位置的概率
        total_prob = 0.0
        for i in range(self.config.board_size):
            for j in range(self.config.board_size):
                if state[i, j] == 0:  # 空位
                    prob = policy_2d[i, j].item()
                    policy_dict[(i, j)] = prob
                    total_prob += prob
        
        # 归一化
        if total_prob > 0:
            for action in policy_dict:
                policy_dict[action] /= total_prob
        
        return policy_dict

class NetworkTrainer:
    """网络训练器"""
    
    def __init__(self, network: AlphaZeroNetwork, learning_rate: float = 0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        
        # 损失函数
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # 训练统计
        self.training_history = []
        
    def train_step(self, states: List[np.ndarray], target_policies: List[Dict], 
                   target_values: List[float]) -> Dict[str, float]:
        """
        执行一步训练
        
        Args:
            states: 棋盘状态列表
            target_policies: 目标策略分布列表
            target_values: 目标价值列表
            
        Returns:
            损失信息
        """
        self.network.train()
        
        # 准备批次数据
        batch_inputs = []
        batch_policy_targets = []
        batch_value_targets = []
        
        for i, state in enumerate(states):
            # 准备输入（假设当前玩家为1）
            input_tensor = self.network._prepare_input(state, 1)
            batch_inputs.append(input_tensor)
            
            # 准备策略目标
            policy_target = torch.zeros(self.network.config.board_size ** 2)
            for (row, col), prob in target_policies[i].items():
                idx = row * self.network.config.board_size + col
                policy_target[idx] = prob
            batch_policy_targets.append(policy_target)
            
            # 准备价值目标
            batch_value_targets.append(target_values[i])
        
        # 转换为张量
        batch_inputs = torch.cat(batch_inputs, dim=0)
        batch_policy_targets = torch.stack(batch_policy_targets)
        batch_value_targets = torch.tensor(batch_value_targets, dtype=torch.float32).unsqueeze(1)
        
        # 前向传播
        policy_pred, value_pred = self.network(batch_inputs)
        
        # 计算损失
        policy_loss = self.policy_loss_fn(policy_pred, batch_policy_targets)
        value_loss = self.value_loss_fn(value_pred, batch_value_targets)
        total_loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 记录损失
        loss_info = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        
        self.training_history.append(loss_info)
        
        return loss_info
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.network.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])

# 特征工程模块
class FeatureExtractor:
    """五子棋特征提取器"""
    
    def __init__(self, board_size: int = 15, history_length: int = 8):
        self.board_size = board_size
        self.history_length = history_length
        
    def extract_features(self, state: np.ndarray, player: int, 
                        move_history: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        提取增强特征
        
        Args:
            state: 当前棋盘状态
            player: 当前玩家
            move_history: 移动历史
            
        Returns:
            特征张量 (channels, height, width)
        """
        features = np.zeros((self.history_length, self.board_size, self.board_size))
        
        # 基础特征
        features[0] = (state == player).astype(np.float32)  # 己方棋子
        features[1] = (state == (3 - player)).astype(np.float32)  # 对方棋子
        features[2] = (state == 0).astype(np.float32)  # 空位
        
        # 边界特征
        features[3] = self._get_boundary_features()
        
        # 威胁特征
        features[4] = self._get_threat_features(state, player)
        features[5] = self._get_threat_features(state, 3 - player)
        
        # 历史移动特征
        if move_history:
            features[6] = self._get_recent_move_features(move_history, -1)  # 最近一步
            features[7] = self._get_recent_move_features(move_history, -2)  # 最近两步
        
        return features
    
    def _get_boundary_features(self) -> np.ndarray:
        """获取边界特征"""
        boundary = np.zeros((self.board_size, self.board_size))
        
        # 边缘位置
        boundary[0, :] = 1  # 上边缘
        boundary[-1, :] = 1  # 下边缘
        boundary[:, 0] = 1  # 左边缘
        boundary[:, -1] = 1  # 右边缘
        
        return boundary
    
    def _get_threat_features(self, state: np.ndarray, player: int) -> np.ndarray:
        """获取威胁特征"""
        threat_map = np.zeros((self.board_size, self.board_size))
        
        # 简化的威胁检测
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i, j] == 0:  # 空位
                    # 模拟落子后的威胁等级
                    temp_state = state.copy()
                    temp_state[i, j] = player
                    threat_level = self._evaluate_threat_at_position(temp_state, i, j, player)
                    threat_map[i, j] = threat_level
        
        return threat_map
    
    def _get_recent_move_features(self, move_history: List[Tuple[int, int]], 
                                 move_index: int) -> np.ndarray:
        """获取最近移动特征"""
        feature_map = np.zeros((self.board_size, self.board_size))
        
        if move_history and abs(move_index) <= len(move_history):
            row, col = move_history[move_index]
            feature_map[row, col] = 1.0
        
        return feature_map
    
    def _evaluate_threat_at_position(self, state: np.ndarray, row: int, col: int, player: int) -> float:
        """评估位置的威胁等级"""
        # 简化的威胁评估
        threat_score = 0.0
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            # 计算该方向的连子数
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while (0 <= x < self.board_size and 0 <= y < self.board_size and 
                   state[x, y] == player):
                count += 1
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while (0 <= x < self.board_size and 0 <= y < self.board_size and 
                   state[x, y] == player):
                count += 1
                x -= dx
                y -= dy
            
            # 根据连子数评分
            if count >= 5:
                threat_score += 1000  # 五连
            elif count == 4:
                threat_score += 100   # 活四或冲四
            elif count == 3:
                threat_score += 10    # 活三或眠三
            elif count == 2:
                threat_score += 1     # 活二或眠二
        
        return min(threat_score / 1000.0, 1.0)  # 归一化到[0,1]

# 测试代码
if __name__ == "__main__":
    # 创建网络配置
    config = NetworkConfig(
        board_size=15,
        input_channels=8,
        residual_blocks=6,
        filters=128
    )
    
    # 创建网络
    network = AlphaZeroNetwork(config)
    
    # 测试前向传播
    test_input = torch.randn(1, 8, 15, 15)
    policy, value = network(test_input)
    
    print(f"网络输出形状:")
    print(f"策略: {policy.shape}")
    print(f"价值: {value.shape}")
    
    # 测试预测接口
    test_state = np.zeros((15, 15))
    test_state[7, 7] = 1
    test_state[7, 8] = 2
    
    policy_dict, value_estimate = network.predict(test_state, 1)
    print(f"\n预测结果:")
    print(f"合法动作数: {len(policy_dict)}")
    print(f"价值估计: {value_estimate:.3f}")
    
    # 测试特征提取
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(test_state, 1)
    print(f"\n特征形状: {features.shape}")
    print(f"特征摘要: {features.sum(axis=(1,2))}")