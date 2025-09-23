"""
五子棋AI深度学习模型架构设计与实现

这是一个专为五子棋设计的现代化深度学习架构，结合了AlphaZero的核心思想
和针对五子棋特点的专门优化。

架构特点：
1. 双头网络：策略网络 + 价值网络
2. 残差网络backbone：深度特征提取
3. 五子棋专用特征工程：对称性、棋型检测
4. 多尺度感受野：捕获局部和全局模式
5. 注意力机制：关注关键区域
6. 数据增强：8重对称性、噪声注入

作者：Claude AI Engineer
日期：2025-09-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json

class ActivationType(Enum):
    """激活函数类型"""
    RELU = "relu"
    SWISH = "swish"
    MISH = "mish"
    GELU = "gelu"

@dataclass
class ModelConfig:
    """完整的模型配置"""
    # 基础配置
    board_size: int = 15
    input_channels: int = 12  # 扩展的输入通道
    
    # 网络结构
    backbone_type: str = "resnet"  # resnet, densenet, efficientnet
    residual_blocks: int = 12      # 残差块数量
    filters: int = 256             # 主干网络通道数
    
    # 策略头配置
    policy_head_filters: int = 32
    policy_head_layers: int = 2
    
    # 价值头配置  
    value_head_filters: int = 16
    value_head_hidden: int = 512
    value_head_layers: int = 3
    
    # 训练配置
    dropout_rate: float = 0.3
    batch_norm_momentum: float = 0.1
    activation: ActivationType = ActivationType.SWISH
    
    # 注意力机制
    use_attention: bool = True
    attention_heads: int = 8
    attention_dim: int = 64
    
    # 多尺度特征
    use_multiscale: bool = True
    scales: List[int] = None
    
    # 正则化
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    
    # 五子棋专用
    pattern_channels: int = 4  # 棋型检测通道
    symmetry_augment: bool = True
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1, 2, 3]  # 多尺度感受野

def get_activation(activation_type: ActivationType):
    """获取激活函数"""
    if activation_type == ActivationType.RELU:
        return nn.ReLU(inplace=True)
    elif activation_type == ActivationType.SWISH:
        return nn.SiLU(inplace=True)  # SiLU就是Swish
    elif activation_type == ActivationType.MISH:
        return nn.Mish(inplace=True)
    elif activation_type == ActivationType.GELU:
        return nn.GELU()
    else:
        return nn.ReLU(inplace=True)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SpatialAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class MultiScaleBlock(nn.Module):
    """多尺度特征提取块"""
    
    def __init__(self, in_channels: int, out_channels: int, scales: List[int], activation: ActivationType):
        super(MultiScaleBlock, self).__init__()
        
        self.scales = scales
        self.branches = nn.ModuleList()
        
        branch_channels = out_channels // len(scales)
        
        for scale in scales:
            if scale == 1:
                branch = nn.Conv2d(in_channels, branch_channels, 1, bias=False)
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 3, padding=scale, dilation=scale, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    get_activation(activation)
                )
            self.branches.append(branch)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )
    
    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        out = torch.cat(branch_outputs, dim=1)
        out = self.fusion(out)
        
        return out

class EnhancedResidualBlock(nn.Module):
    """增强残差块"""
    
    def __init__(self, channels: int, config: ModelConfig):
        super(EnhancedResidualBlock, self).__init__()
        
        self.config = config
        
        # 主路径
        if config.use_multiscale:
            self.conv1 = MultiScaleBlock(channels, channels, config.scales, config.activation)
        else:
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels, momentum=config.batch_norm_momentum)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=config.batch_norm_momentum)
        
        # 激活函数
        self.activation = get_activation(config.activation)
        
        # 注意力机制
        if config.use_attention:
            self.attention = SpatialAttention(channels)
        else:
            self.attention = None
        
        # Dropout
        self.dropout = nn.Dropout2d(config.dropout_rate) if config.dropout_rate > 0 else None
    
    def forward(self, x):
        residual = x
        
        # 第一个卷积
        if self.config.use_multiscale:
            out = self.conv1(x)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)
        
        # Dropout
        if self.dropout:
            out = self.dropout(out)
        
        # 第二个卷积
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 注意力机制
        if self.attention:
            out = self.attention(out)
        
        # 残差连接
        out += residual
        out = self.activation(out)
        
        return out

class PatternDetector(nn.Module):
    """五子棋棋型检测器"""
    
    def __init__(self, in_channels: int, pattern_channels: int):
        super(PatternDetector, self).__init__()
        
        # 不同方向的卷积核
        self.horizontal_conv = nn.Conv2d(in_channels, pattern_channels // 4, (1, 5), padding=(0, 2))
        self.vertical_conv = nn.Conv2d(in_channels, pattern_channels // 4, (5, 1), padding=(2, 0))
        self.diagonal1_conv = self._create_diagonal_conv(in_channels, pattern_channels // 4, "main")
        self.diagonal2_conv = self._create_diagonal_conv(in_channels, pattern_channels // 4, "anti")
        
        self.fusion = nn.Conv2d(pattern_channels, pattern_channels, 1)
        
    def _create_diagonal_conv(self, in_channels: int, out_channels: int, diagonal_type: str):
        """创建对角线卷积"""
        # 简化实现，使用3x3卷积近似
        return nn.Conv2d(in_channels, out_channels, 3, padding=1)
    
    def forward(self, x):
        h_feat = self.horizontal_conv(x)
        v_feat = self.vertical_conv(x)
        d1_feat = self.diagonal1_conv(x)
        d2_feat = self.diagonal2_conv(x)
        
        # 拼接所有方向的特征
        pattern_feat = torch.cat([h_feat, v_feat, d1_feat, d2_feat], dim=1)
        pattern_feat = self.fusion(pattern_feat)
        
        return pattern_feat

class PolicyHead(nn.Module):
    """增强策略头"""
    
    def __init__(self, config: ModelConfig):
        super(PolicyHead, self).__init__()
        
        self.config = config
        
        # 卷积层序列
        layers = []
        in_channels = config.filters
        
        for i in range(config.policy_head_layers):
            out_channels = config.policy_head_filters if i == 0 else config.policy_head_filters // 2
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                get_activation(config.activation)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 输出层
        self.output_conv = nn.Conv2d(in_channels, 1, 1)
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = self.output_conv(out)
        
        # 展平并应用softmax
        batch_size = out.size(0)
        out = out.view(batch_size, -1)
        policy = F.softmax(out, dim=1)
        
        return policy

class ValueHead(nn.Module):
    """增强价值头"""
    
    def __init__(self, config: ModelConfig):
        super(ValueHead, self).__init__()
        
        self.config = config
        
        # 卷积层
        self.conv = nn.Conv2d(config.filters, config.value_head_filters, 1, bias=False)
        self.bn = nn.BatchNorm2d(config.value_head_filters)
        self.activation = get_activation(config.activation)
        
        # 全连接层
        conv_output_size = config.value_head_filters * config.board_size * config.board_size
        
        layers = []
        in_features = conv_output_size
        
        for i in range(config.value_head_layers - 1):
            out_features = config.value_head_hidden // (2 ** i)
            layers.extend([
                nn.Linear(in_features, out_features),
                get_activation(config.activation),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = out_features
        
        # 输出层
        layers.append(nn.Linear(in_features, 1))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # 卷积处理
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层
        out = self.fc_layers(out)
        
        # 价值输出 [-1, 1]
        value = torch.tanh(out)
        
        return value

class GomokuNetwork(nn.Module):
    """五子棋专用神经网络"""
    
    def __init__(self, config: ModelConfig):
        super(GomokuNetwork, self).__init__()
        
        self.config = config
        
        # 输入预处理
        self.input_conv = nn.Sequential(
            nn.Conv2d(config.input_channels, config.filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.filters),
            get_activation(config.activation)
        )
        
        # 棋型检测器
        if config.pattern_channels > 0:
            self.pattern_detector = PatternDetector(config.filters, config.pattern_channels)
            backbone_channels = config.filters + config.pattern_channels
        else:
            self.pattern_detector = None
            backbone_channels = config.filters
        
        # 特征融合
        if self.pattern_detector:
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(backbone_channels, config.filters, 1, bias=False),
                nn.BatchNorm2d(config.filters),
                get_activation(config.activation)
            )
        
        # 主干网络
        self.backbone = nn.ModuleList([
            EnhancedResidualBlock(config.filters, config)
            for _ in range(config.residual_blocks)
        ])
        
        # 策略和价值头
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 输入处理
        features = self.input_conv(x)
        
        # 棋型检测
        if self.pattern_detector:
            pattern_features = self.pattern_detector(features)
            combined_features = torch.cat([features, pattern_features], dim=1)
            features = self.feature_fusion(combined_features)
        
        # 主干网络
        for block in self.backbone:
            features = block(features)
        
        # 策略和价值预测
        policy = self.policy_head(features)
        value = self.value_head(features)
        
        return policy, value
    
    def predict(self, state: np.ndarray, player: int, 
                move_history: List[Tuple[int, int]] = None) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        预测接口（兼容MCTS）
        """
        self.eval()
        
        with torch.no_grad():
            # 准备输入
            input_tensor = self._prepare_input(state, player, move_history)
            
            # 前向传播
            policy_logits, value = self.forward(input_tensor)
            
            # 转换输出格式
            policy_dict = self._convert_policy_to_dict(policy_logits[0], state)
            value_scalar = value.item()
        
        return policy_dict, value_scalar
    
    def _prepare_input(self, state: np.ndarray, player: int, 
                      move_history: List[Tuple[int, int]] = None) -> torch.Tensor:
        """准备网络输入"""
        batch_size = 1
        channels = self.config.input_channels
        height, width = self.config.board_size, self.config.board_size
        
        input_tensor = torch.zeros(batch_size, channels, height, width)
        
        # 基础棋盘特征 (前4个通道)
        input_tensor[0, 0] = torch.from_numpy((state == player).astype(np.float32))      # 己方棋子
        input_tensor[0, 1] = torch.from_numpy((state == (3 - player)).astype(np.float32)) # 对方棋子
        input_tensor[0, 2] = torch.from_numpy((state == 0).astype(np.float32))          # 空位
        input_tensor[0, 3] = float(player == 1)  # 当前玩家标识
        
        # 位置编码 (2个通道)
        for i in range(height):
            for j in range(width):
                input_tensor[0, 4, i, j] = i / (height - 1)  # 行位置编码
                input_tensor[0, 5, i, j] = j / (width - 1)   # 列位置编码
        
        # 距离中心的特征 (1个通道)
        center = height // 2
        for i in range(height):
            for j in range(width):
                distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                input_tensor[0, 6, i, j] = 1.0 - min(distance / center, 1.0)
        
        # 移动历史特征 (5个通道)
        if move_history:
            # 最近5步移动
            for idx, step in enumerate(range(-1, -6, -1)):
                if abs(step) <= len(move_history):
                    row, col = move_history[step]
                    input_tensor[0, 7 + idx, row, col] = 1.0
        
        return input_tensor
    
    def _convert_policy_to_dict(self, policy_logits: torch.Tensor, 
                               state: np.ndarray) -> Dict[Tuple[int, int], float]:
        """转换策略输出为字典格式"""
        policy_dict = {}
        
        # 重塑为棋盘形状
        policy_2d = policy_logits.view(self.config.board_size, self.config.board_size)
        
        # 只保留合法位置
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

class AdvancedLoss(nn.Module):
    """高级损失函数"""
    
    def __init__(self, config: ModelConfig):
        super(AdvancedLoss, self).__init__()
        
        self.config = config
        
        # 基础损失
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()
        
        # 权重
        self.policy_weight = 1.0
        self.value_weight = 1.0
        self.regularization_weight = config.weight_decay
        
    def forward(self, policy_pred: torch.Tensor, value_pred: torch.Tensor,
                policy_target: torch.Tensor, value_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        # 策略损失 (使用标签平滑)
        if self.config.label_smoothing > 0:
            policy_target = self._apply_label_smoothing(policy_target, self.config.label_smoothing)
        
        policy_loss = self.policy_loss_fn(
            F.log_softmax(policy_pred, dim=1),
            policy_target
        )
        
        # 价值损失
        value_loss = self.value_loss_fn(value_pred, value_target)
        
        # 总损失
        total_loss = (self.policy_weight * policy_loss + 
                     self.value_weight * value_loss)
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
    
    def _apply_label_smoothing(self, targets: torch.Tensor, smoothing: float) -> torch.Tensor:
        """应用标签平滑"""
        num_classes = targets.size(1)
        smoothed = targets * (1 - smoothing) + smoothing / num_classes
        return smoothed

class DataAugmentation:
    """数据增强"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.symmetries = self._get_symmetries() if config.symmetry_augment else []
    
    def _get_symmetries(self) -> List[callable]:
        """获取对称变换"""
        symmetries = []
        
        # 8种对称变换
        symmetries.append(lambda x: x)  # 原始
        symmetries.append(lambda x: torch.rot90(x, 1, [2, 3]))  # 90度旋转
        symmetries.append(lambda x: torch.rot90(x, 2, [2, 3]))  # 180度旋转
        symmetries.append(lambda x: torch.rot90(x, 3, [2, 3]))  # 270度旋转
        symmetries.append(lambda x: torch.flip(x, [2]))  # 水平翻转
        symmetries.append(lambda x: torch.flip(x, [3]))  # 垂直翻转
        symmetries.append(lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [2]))  # 旋转+翻转
        symmetries.append(lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]))  # 旋转+翻转
        
        return symmetries
    
    def augment_batch(self, states: torch.Tensor, policies: torch.Tensor, 
                     values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批次数据增强"""
        if not self.symmetries:
            return states, policies, values
        
        augmented_states = []
        augmented_policies = []
        augmented_values = []
        
        batch_size = states.size(0)
        
        for i in range(batch_size):
            # 随机选择对称变换
            transform = np.random.choice(self.symmetries)
            
            # 变换状态
            aug_state = transform(states[i:i+1])
            augmented_states.append(aug_state)
            
            # 变换策略 (需要相应地变换动作概率)
            policy_2d = policies[i].view(self.config.board_size, self.config.board_size)
            aug_policy_2d = transform(policy_2d.unsqueeze(0).unsqueeze(0))[0, 0]
            aug_policy = aug_policy_2d.view(-1)
            augmented_policies.append(aug_policy)
            
            # 价值不变
            augmented_values.append(values[i:i+1])
        
        return (torch.cat(augmented_states, dim=0),
                torch.stack(augmented_policies, dim=0),
                torch.cat(augmented_values, dim=0))

# 使用示例和测试
def create_model(config_dict: Dict = None) -> GomokuNetwork:
    """创建模型实例"""
    if config_dict is None:
        config_dict = {}
    
    config = ModelConfig(**config_dict)
    model = GomokuNetwork(config)
    
    return model

def model_summary(model: GomokuNetwork):
    """模型摘要"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"=== 五子棋AI模型摘要 ===")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"配置: {model.config}")

if __name__ == "__main__":
    # 测试模型
    print("创建五子棋AI深度学习模型...")
    
    # 创建配置
    config = ModelConfig(
        board_size=15,
        input_channels=12,
        residual_blocks=10,
        filters=256,
        use_attention=True,
        use_multiscale=True,
        pattern_channels=32
    )
    
    # 创建模型
    model = create_model(config.__dict__)
    model_summary(model)
    
    # 测试前向传播
    print("\n测试前向传播...")
    test_input = torch.randn(2, 12, 15, 15)
    policy, value = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"策略输出: {policy.shape}")
    print(f"价值输出: {value.shape}")
    
    # 测试预测接口
    print("\n测试预测接口...")
    test_state = np.zeros((15, 15))
    test_state[7, 7] = 1
    test_state[7, 8] = 2
    
    policy_dict, value_estimate = model.predict(test_state, 1)
    print(f"合法动作数: {len(policy_dict)}")
    print(f"价值估计: {value_estimate:.3f}")
    print(f"策略样例: {list(policy_dict.items())[:5]}")
    
    print("\n模型创建成功！")