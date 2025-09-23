"""
针对五子棋优化的残差网络架构
包含特殊的卷积层设计和注意力机制

作者：Claude AI Engineer
日期：2025-09-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import math

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # 融合并应用sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out)
        
        return x * attention

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道方向的平均和最大
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接并卷积
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        
        return x * attention

class CBAM(nn.Module):
    """卷积块注意力模块（Convolutional Block Attention Module）"""
    
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EnhancedResidualBlock(nn.Module):
    """增强的残差块"""
    
    def __init__(self, channels: int, use_attention: bool = True, dropout_rate: float = 0.1):
        super(EnhancedResidualBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 注意力机制
        self.attention = CBAM(channels) if use_attention else None
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout:
            out = self.dropout(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 注意力机制
        if self.attention:
            out = self.attention(out)
        
        # 残差连接
        out += residual
        out = self.relu(out)
        
        return out

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 不同尺度的卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        
        # 最大池化分支
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        
        # 批归一化
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 不同尺度的特征
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(x)
        branch3 = self.conv5x5(x)
        
        # 池化分支
        branch4 = self.maxpool(x)
        branch4 = self.pool_conv(branch4)
        
        # 拼接所有分支
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class GomokuSpecificResNet(nn.Module):
    """五子棋专用残差网络"""
    
    def __init__(self, 
                 board_size: int = 15,
                 input_channels: int = 8,
                 base_filters: int = 256,
                 num_residual_blocks: int = 12,
                 use_attention: bool = True,
                 use_multiscale: bool = True):
        super(GomokuSpecificResNet, self).__init__()
        
        self.board_size = board_size
        self.base_filters = base_filters
        
        # 输入处理层
        if use_multiscale:
            self.input_layer = MultiScaleFeatureExtractor(input_channels, base_filters)
        else:
            self.input_layer = nn.Sequential(
                nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_filters),
                nn.ReLU(inplace=True)
            )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            EnhancedResidualBlock(base_filters, use_attention=use_attention)
            for _ in range(num_residual_blocks)
        ])
        
        # 策略头
        self.policy_head = self._build_policy_head()
        
        # 价值头
        self.value_head = self._build_value_head()
        
        # 初始化权重
        self._initialize_weights()
    
    def _build_policy_head(self):
        """构建策略头"""
        return nn.Sequential(
            nn.Conv2d(self.base_filters, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, self.board_size * self.board_size)
        )
    
    def _build_value_head(self):
        """构建价值头"""
        return nn.Sequential(
            nn.Conv2d(self.base_filters, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入处理
        x = self.input_layer(x)
        
        # 通过残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 策略和价值预测
        policy = F.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        
        return policy, value

class SymmetryAugmentation:
    """对称性数据增强"""
    
    @staticmethod
    def get_all_symmetries(state: np.ndarray, policy: Dict[Tuple[int, int], float]) -> List[Tuple[np.ndarray, Dict]]:
        """
        获取所有8种对称变换
        
        Args:
            state: 棋盘状态
            policy: 策略分布
            
        Returns:
            所有对称变换的(状态, 策略)对
        """
        symmetries = []
        
        # 原始状态
        symmetries.append((state.copy(), policy.copy()))
        
        # 90度旋转
        for k in range(1, 4):
            rotated_state = np.rot90(state, k)
            rotated_policy = SymmetryAugmentation._rotate_policy(policy, state.shape[0], k)
            symmetries.append((rotated_state, rotated_policy))
        
        # 水平翻转
        flipped_state = np.fliplr(state)
        flipped_policy = SymmetryAugmentation._flip_policy_horizontal(policy, state.shape[0])
        symmetries.append((flipped_state, flipped_policy))
        
        # 水平翻转 + 旋转
        for k in range(1, 4):
            rotated_flipped_state = np.rot90(flipped_state, k)
            rotated_flipped_policy = SymmetryAugmentation._rotate_policy(flipped_policy, state.shape[0], k)
            symmetries.append((rotated_flipped_state, rotated_flipped_policy))
        
        return symmetries
    
    @staticmethod
    def _rotate_policy(policy: Dict[Tuple[int, int], float], board_size: int, k: int) -> Dict[Tuple[int, int], float]:
        """旋转策略分布"""
        rotated_policy = {}
        
        for (row, col), prob in policy.items():
            # 旋转坐标
            for _ in range(k):
                row, col = col, board_size - 1 - row
            rotated_policy[(row, col)] = prob
        
        return rotated_policy
    
    @staticmethod
    def _flip_policy_horizontal(policy: Dict[Tuple[int, int], float], board_size: int) -> Dict[Tuple[int, int], float]:
        """水平翻转策略分布"""
        flipped_policy = {}
        
        for (row, col), prob in policy.items():
            flipped_col = board_size - 1 - col
            flipped_policy[(row, flipped_col)] = prob
        
        return flipped_policy

class GomokuPatternDetector(nn.Module):
    """五子棋模式检测器"""
    
    def __init__(self, channels: int):
        super(GomokuPatternDetector, self).__init__()
        
        # 定义五子棋特有的卷积核
        self.pattern_convs = nn.ModuleList([
            # 水平模式
            nn.Conv2d(channels, 16, kernel_size=(1, 5), padding=(0, 2)),
            # 垂直模式
            nn.Conv2d(channels, 16, kernel_size=(5, 1), padding=(2, 0)),
            # 对角线模式
            nn.Conv2d(channels, 16, kernel_size=5, padding=2),
            # 反对角线模式（需要特殊处理）
            nn.Conv2d(channels, 16, kernel_size=5, padding=2)
        ])
        
        # 融合层
        self.fusion_conv = nn.Conv2d(64, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化对角线卷积核
        self._initialize_diagonal_kernels()
    
    def _initialize_diagonal_kernels(self):
        """初始化对角线检测卷积核"""
        with torch.no_grad():
            # 主对角线卷积核
            diag_kernel = torch.zeros(5, 5)
            for i in range(5):
                diag_kernel[i, i] = 1.0
            
            # 反对角线卷积核
            anti_diag_kernel = torch.zeros(5, 5)
            for i in range(5):
                anti_diag_kernel[i, 4-i] = 1.0
            
            # 设置卷积核权重
            for i in range(16):
                self.pattern_convs[2].weight[i, 0] = diag_kernel
                self.pattern_convs[3].weight[i, 0] = anti_diag_kernel
    
    def forward(self, x):
        # 提取不同方向的模式
        pattern_features = []
        for conv in self.pattern_convs:
            feature = conv(x)
            pattern_features.append(feature)
        
        # 拼接所有模式特征
        combined = torch.cat(pattern_features, dim=1)
        
        # 融合
        out = self.fusion_conv(combined)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class AdvancedGomokuNetwork(nn.Module):
    """高级五子棋网络（集成所有优化技术）"""
    
    def __init__(self, config_dict: Dict):
        super(AdvancedGomokuNetwork, self).__init__()
        
        # 解析配置
        self.board_size = config_dict.get('board_size', 15)
        input_channels = config_dict.get('input_channels', 8)
        base_filters = config_dict.get('base_filters', 256)
        num_blocks = config_dict.get('num_residual_blocks', 12)
        
        # 输入处理
        self.input_conv = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(base_filters)
        
        # 模式检测器
        self.pattern_detector = GomokuPatternDetector(base_filters)
        
        # 残差网络主干
        self.backbone = GomokuSpecificResNet(
            board_size=self.board_size,
            input_channels=base_filters,  # 注意这里传入的是处理后的通道数
            base_filters=base_filters,
            num_residual_blocks=num_blocks,
            use_attention=True,
            use_multiscale=False  # 已经在输入层使用了
        )
        
    def forward(self, x):
        # 输入处理
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # 模式检测
        x = self.pattern_detector(x)
        
        # 主干网络
        policy, value = self.backbone(x)
        
        return policy, value

# 性能测试和对比
def performance_comparison():
    """性能对比测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 标准配置
    config = {
        'board_size': 15,
        'input_channels': 8,
        'base_filters': 256,
        'num_residual_blocks': 10
    }
    
    # 创建不同网络
    networks = {
        'basic_resnet': GomokuSpecificResNet(use_attention=False, use_multiscale=False),
        'resnet_with_attention': GomokuSpecificResNet(use_attention=True, use_multiscale=False),
        'resnet_with_multiscale': GomokuSpecificResNet(use_attention=False, use_multiscale=True),
        'advanced_network': AdvancedGomokuNetwork(config)
    }
    
    # 测试输入
    test_input = torch.randn(1, 8, 15, 15).to(device)
    
    print("网络性能对比:")
    print("-" * 60)
    
    for name, network in networks.items():
        network = network.to(device)
        network.eval()
        
        # 计算参数数量
        total_params = sum(p.numel() for p in network.parameters())
        
        # 计算推理时间
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            # 预热
            for _ in range(10):
                _ = network(test_input)
            
            # 正式测试
            start_time.record()
            for _ in range(100):
                policy, value = network(test_input)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 100  # 平均时间
        
        print(f"{name:20} | 参数: {total_params:8d} | 推理时间: {elapsed_time:.2f}ms")

if __name__ == "__main__":
    # 创建高级网络
    config = {
        'board_size': 15,
        'input_channels': 8,
        'base_filters': 256,
        'num_residual_blocks': 12
    }
    
    network = AdvancedGomokuNetwork(config)
    
    # 测试前向传播
    test_input = torch.randn(2, 8, 15, 15)
    policy, value = network(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"策略输出形状: {policy.shape}")
    print(f"价值输出形状: {value.shape}")
    
    # 测试对称性增强
    test_state = np.random.randint(0, 3, (15, 15))
    test_policy = {(7, 7): 0.5, (7, 8): 0.3, (8, 7): 0.2}
    
    symmetries = SymmetryAugmentation.get_all_symmetries(test_state, test_policy)
    print(f"\n对称性增强结果: {len(symmetries)} 个变换")
    
    # 性能对比（需要GPU）
    if torch.cuda.is_available():
        print("\n开始性能测试...")
        performance_comparison()
    else:
        print("\n需要GPU进行性能测试")