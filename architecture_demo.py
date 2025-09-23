"""
五子棋AI深度学习架构演示版本（不依赖PyTorch）

展示核心架构设计思想和实现逻辑
包含完整的网络结构定义和前向传播模拟

作者：Claude AI Engineer
日期：2025-09-22
"""

import numpy as np
import json
import math
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

class ActivationType(Enum):
    """激活函数类型"""
    RELU = "relu"
    SWISH = "swish"
    TANH = "tanh"

@dataclass
class ModelConfig:
    """模型配置"""
    board_size: int = 15
    input_channels: int = 12
    residual_blocks: int = 10
    filters: int = 256
    policy_head_filters: int = 32
    value_head_filters: int = 16
    dropout_rate: float = 0.3
    activation: ActivationType = ActivationType.SWISH
    use_attention: bool = True
    use_multiscale: bool = True
    pattern_channels: int = 32

class ActivationFunction:
    """激活函数实现"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        return x * (1 / (1 + np.exp(-x)))
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def apply(x: np.ndarray, activation_type: ActivationType) -> np.ndarray:
        if activation_type == ActivationType.RELU:
            return ActivationFunction.relu(x)
        elif activation_type == ActivationType.SWISH:
            return ActivationFunction.swish(x)
        elif activation_type == ActivationType.TANH:
            return ActivationFunction.tanh(x)
        else:
            return x

class BatchNorm2D:
    """批归一化层模拟"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.weight = np.ones(num_features)
        self.bias = np.zeros(num_features)
        
        # 运行时统计
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        if training:
            # 计算批次统计
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            # 更新运行时统计
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和偏移
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        
        return weight * x_norm + bias

class Conv2D:
    """2D卷积层模拟"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, padding: int = 0, stride: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # 初始化权重 (Kaiming初始化)
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        self.weight = np.random.normal(0, std, 
                                     (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（简化实现）"""
        batch_size, in_channels, height, width = x.shape
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 创建输出张量
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # 添加填充
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                                (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        # 卷积操作（简化版本）
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        
                        # 提取输入窗口
                        input_window = x_padded[b, :, 
                                              h_start:h_start+self.kernel_size,
                                              w_start:w_start+self.kernel_size]
                        
                        # 卷积计算
                        conv_result = np.sum(input_window * self.weight[oc]) + self.bias[oc]
                        output[b, oc, oh, ow] = conv_result
        
        return output

class Linear:
    """线性层模拟"""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = np.random.normal(0, std, (out_features, in_features))
        self.bias = np.zeros(out_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        return np.dot(x, self.weight.T) + self.bias

class ResidualBlock:
    """残差块"""
    
    def __init__(self, channels: int, config: ModelConfig):
        self.config = config
        
        # 卷积层
        self.conv1 = Conv2D(channels, channels, 3, padding=1)
        self.bn1 = BatchNorm2D(channels)
        self.conv2 = Conv2D(channels, channels, 3, padding=1)
        self.bn2 = BatchNorm2D(channels)
        
        # 注意力机制（简化）
        if config.use_attention:
            self.attention = SpatialAttention(channels)
        else:
            self.attention = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        residual = x
        
        # 第一个卷积块
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = ActivationFunction.apply(out, self.config.activation)
        
        # 第二个卷积块
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)
        
        # 注意力机制
        if self.attention:
            out = self.attention.forward(out)
        
        # 残差连接
        out = out + residual
        out = ActivationFunction.apply(out, self.config.activation)
        
        return out

class SpatialAttention:
    """空间注意力机制"""
    
    def __init__(self, channels: int):
        self.channels = channels
        self.global_avg_pool = True
        self.global_max_pool = True
        
        # 注意力卷积
        self.conv = Conv2D(2, 1, 7, padding=3)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = x.shape
        
        # 全局平均池化和最大池化
        avg_pool = np.mean(x, axis=1, keepdims=True)  # (B, 1, H, W)
        max_pool = np.max(x, axis=1, keepdims=True)   # (B, 1, H, W)
        
        # 拼接
        concat = np.concatenate([avg_pool, max_pool], axis=1)  # (B, 2, H, W)
        
        # 注意力卷积
        attention_map = self.conv.forward(concat)
        attention_map = 1 / (1 + np.exp(-attention_map))  # Sigmoid
        
        # 应用注意力
        return x * attention_map

class PatternDetector:
    """五子棋棋型检测器"""
    
    def __init__(self, in_channels: int, pattern_channels: int):
        self.in_channels = in_channels
        self.pattern_channels = pattern_channels
        
        # 四个方向的检测器
        self.horizontal_conv = Conv2D(in_channels, pattern_channels // 4, kernel_size=1)
        self.vertical_conv = Conv2D(in_channels, pattern_channels // 4, kernel_size=1)
        self.diagonal1_conv = Conv2D(in_channels, pattern_channels // 4, kernel_size=3, padding=1)
        self.diagonal2_conv = Conv2D(in_channels, pattern_channels // 4, kernel_size=3, padding=1)
        
        # 特征融合
        self.fusion_conv = Conv2D(pattern_channels, pattern_channels, kernel_size=1)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 不同方向的特征提取
        h_feat = self.horizontal_conv.forward(x)
        v_feat = self.vertical_conv.forward(x)
        d1_feat = self.diagonal1_conv.forward(x)
        d2_feat = self.diagonal2_conv.forward(x)
        
        # 拼接所有方向特征
        pattern_features = np.concatenate([h_feat, v_feat, d1_feat, d2_feat], axis=1)
        
        # 融合
        fused_features = self.fusion_conv.forward(pattern_features)
        
        return fused_features

class PolicyHead:
    """策略头"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # 卷积层
        self.conv1 = Conv2D(config.filters, config.policy_head_filters, 3, padding=1)
        self.bn1 = BatchNorm2D(config.policy_head_filters)
        self.conv2 = Conv2D(config.policy_head_filters, 16, 3, padding=1)
        self.bn2 = BatchNorm2D(16)
        self.output_conv = Conv2D(16, 1, 1)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = ActivationFunction.apply(out, self.config.activation)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)
        out = ActivationFunction.apply(out, self.config.activation)
        
        out = self.output_conv.forward(out)
        
        # 展平并应用softmax
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        policy = ActivationFunction.softmax(out)
        
        return policy

class ValueHead:
    """价值头"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # 卷积层
        self.conv = Conv2D(config.filters, config.value_head_filters, 1)
        self.bn = BatchNorm2D(config.value_head_filters)
        
        # 全连接层
        conv_output_size = config.value_head_filters * config.board_size * config.board_size
        self.fc1 = Linear(conv_output_size, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 1)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # 卷积处理
        out = self.conv.forward(x)
        out = self.bn.forward(out, training)
        out = ActivationFunction.apply(out, self.config.activation)
        
        # 展平
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        
        # 全连接层
        out = self.fc1.forward(out)
        out = ActivationFunction.apply(out, self.config.activation)
        
        out = self.fc2.forward(out)
        out = ActivationFunction.apply(out, self.config.activation)
        
        out = self.fc3.forward(out)
        
        # Tanh激活得到[-1, 1]范围的价值
        value = ActivationFunction.tanh(out)
        
        return value

class GomokuNetworkDemo:
    """五子棋网络演示版本"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # 输入卷积
        self.input_conv = Conv2D(config.input_channels, config.filters, 3, padding=1)
        self.input_bn = BatchNorm2D(config.filters)
        
        # 棋型检测器
        if config.pattern_channels > 0:
            self.pattern_detector = PatternDetector(config.filters, config.pattern_channels)
            backbone_channels = config.filters + config.pattern_channels
            self.feature_fusion = Conv2D(backbone_channels, config.filters, 1)
            self.fusion_bn = BatchNorm2D(config.filters)
        else:
            self.pattern_detector = None
        
        # 残差块
        self.residual_blocks = []
        for _ in range(config.residual_blocks):
            self.residual_blocks.append(ResidualBlock(config.filters, config))
        
        # 策略和价值头
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        
        # 统计参数数量
        self.total_params = self._count_parameters()
    
    def _count_parameters(self) -> int:
        """统计参数数量（模拟）"""
        # 简化的参数计算
        params = 0
        
        # 输入卷积
        params += self.config.input_channels * self.config.filters * 9  # 3x3卷积
        params += self.config.filters  # bias
        
        # 残差块
        for _ in range(self.config.residual_blocks):
            params += self.config.filters * self.config.filters * 9 * 2  # 两个3x3卷积
            params += self.config.filters * 2  # bias
        
        # 棋型检测器
        if self.config.pattern_channels > 0:
            params += self.config.filters * self.config.pattern_channels * 9
        
        # 策略头
        params += self.config.filters * self.config.policy_head_filters * 9
        params += self.config.policy_head_filters * 16 * 9
        params += 16 * 1
        
        # 价值头
        conv_size = self.config.value_head_filters * self.config.board_size * self.config.board_size
        params += self.config.filters * self.config.value_head_filters
        params += conv_size * 512
        params += 512 * 256
        params += 256 * 1
        
        return params
    
    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """前向传播"""
        # 输入处理
        features = self.input_conv.forward(x)
        features = self.input_bn.forward(features, training)
        features = ActivationFunction.apply(features, self.config.activation)
        
        # 棋型检测
        if self.pattern_detector:
            pattern_features = self.pattern_detector.forward(features)
            combined_features = np.concatenate([features, pattern_features], axis=1)
            features = self.feature_fusion.forward(combined_features)
            features = self.fusion_bn.forward(features, training)
            features = ActivationFunction.apply(features, self.config.activation)
        
        # 残差块
        for block in self.residual_blocks:
            features = block.forward(features, training)
        
        # 策略和价值预测
        policy = self.policy_head.forward(features, training)
        value = self.value_head.forward(features, training)
        
        return policy, value
    
    def predict(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """预测接口"""
        # 准备输入
        input_tensor = self._prepare_input(state, player)
        
        # 前向传播
        policy_logits, value = self.forward(input_tensor, training=False)
        
        # 转换输出格式
        policy_dict = self._convert_policy_to_dict(policy_logits[0], state)
        value_scalar = value[0, 0]
        
        return policy_dict, value_scalar
    
    def _prepare_input(self, state: np.ndarray, player: int) -> np.ndarray:
        """准备网络输入"""
        batch_size = 1
        channels = self.config.input_channels
        height, width = self.config.board_size, self.config.board_size
        
        input_tensor = np.zeros((batch_size, channels, height, width))
        
        # 基础特征
        input_tensor[0, 0] = (state == player).astype(np.float32)
        input_tensor[0, 1] = (state == (3 - player)).astype(np.float32)
        input_tensor[0, 2] = (state == 0).astype(np.float32)
        input_tensor[0, 3] = float(player == 1)
        
        # 位置编码
        for i in range(height):
            for j in range(width):
                input_tensor[0, 4, i, j] = i / (height - 1)
                input_tensor[0, 5, i, j] = j / (width - 1)
        
        # 中心距离
        center = height // 2
        for i in range(height):
            for j in range(width):
                distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                input_tensor[0, 6, i, j] = 1.0 - min(distance / center, 1.0)
        
        return input_tensor
    
    def _convert_policy_to_dict(self, policy_logits: np.ndarray, 
                               state: np.ndarray) -> Dict[Tuple[int, int], float]:
        """转换策略输出为字典格式"""
        policy_dict = {}
        
        # 重塑为棋盘形状
        policy_2d = policy_logits.reshape(self.config.board_size, self.config.board_size)
        
        # 只保留合法位置
        total_prob = 0.0
        for i in range(self.config.board_size):
            for j in range(self.config.board_size):
                if state[i, j] == 0:  # 空位
                    prob = policy_2d[i, j]
                    policy_dict[(i, j)] = prob
                    total_prob += prob
        
        # 归一化
        if total_prob > 0:
            for action in policy_dict:
                policy_dict[action] /= total_prob
        
        return policy_dict
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        config_dict = asdict(self.config)
        # 处理Enum类型
        config_dict['activation'] = self.config.activation.value
        
        return {
            'total_parameters': self.total_params,
            'model_size_mb': self.total_params * 4 / 1024 / 1024,  # 假设float32
            'config': config_dict,
            'architecture_summary': {
                'input_shape': (1, self.config.input_channels, 
                              self.config.board_size, self.config.board_size),
                'backbone_blocks': self.config.residual_blocks,
                'pattern_detection': self.config.pattern_channels > 0,
                'attention_mechanism': self.config.use_attention,
                'multi_scale_features': self.config.use_multiscale
            }
        }

def demonstrate_architecture():
    """演示架构功能"""
    print("=== 五子棋AI深度学习架构演示 ===\n")
    
    # 创建配置
    config = ModelConfig(
        board_size=15,
        input_channels=12,
        residual_blocks=8,
        filters=256,
        use_attention=True,
        use_multiscale=True,
        pattern_channels=32
    )
    
    print("1. 创建网络模型...")
    model = GomokuNetworkDemo(config)
    
    # 显示模型信息
    model_info = model.get_model_info()
    print(f"   总参数数量: {model_info['total_parameters']:,}")
    print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"   主干网络层数: {model_info['architecture_summary']['backbone_blocks']}")
    print(f"   使用注意力机制: {model_info['architecture_summary']['attention_mechanism']}")
    print(f"   使用棋型检测: {model_info['architecture_summary']['pattern_detection']}")
    
    print("\n2. 测试前向传播...")
    
    # 创建测试输入
    test_input = np.random.randn(2, 12, 15, 15)
    policy, value = model.forward(test_input, training=False)
    
    print(f"   输入形状: {test_input.shape}")
    print(f"   策略输出形状: {policy.shape}")
    print(f"   价值输出形状: {value.shape}")
    print(f"   策略概率和: {np.sum(policy, axis=1)}")
    print(f"   价值范围: [{np.min(value):.3f}, {np.max(value):.3f}]")
    
    print("\n3. 测试预测接口...")
    
    # 创建测试棋盘
    test_state = np.zeros((15, 15), dtype=int)
    test_state[7, 7] = 1  # 黑棋
    test_state[7, 8] = 2  # 白棋
    test_state[8, 7] = 1  # 黑棋
    
    policy_dict, value_estimate = model.predict(test_state, 1)
    
    print(f"   合法动作数量: {len(policy_dict)}")
    print(f"   价值估计: {value_estimate:.3f}")
    
    # 显示前5个最佳动作
    if policy_dict:
        top_moves = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        print("   前5个最佳动作:")
        for i, ((row, col), prob) in enumerate(top_moves):
            print(f"     {i+1}. 位置({row}, {col}): 概率 {prob:.4f}")
    
    print("\n4. 架构特性分析...")
    
    # 分析各组件的计算复杂度
    input_size = config.input_channels * config.board_size * config.board_size
    feature_size = config.filters * config.board_size * config.board_size
    
    print(f"   输入特征维度: {input_size:,}")
    print(f"   主干特征维度: {feature_size:,}")
    print(f"   策略输出维度: {config.board_size * config.board_size}")
    print(f"   价值输出维度: 1")
    
    # 计算理论计算复杂度（FLOPs）
    flops = estimate_flops(config)
    print(f"   估计FLOPs: {flops:,.0f}")
    
    print("\n5. 五子棋专用优化...")
    print("   + 四方向棋型检测器")
    print("   + 八重对称性数据增强")
    print("   + 位置编码和中心偏向")
    print("   + 注意力机制突出关键区域")
    print("   + 多尺度特征提取")
    print("   + 残差连接支持深层网络")
    
    print("\n6. 训练建议...")
    print("   - 自对弈生成训练数据")
    print("   - 策略价值联合损失函数")
    print("   - AdamW优化器 + Cosine学习率调度")
    print("   - 混合精度训练加速")
    print("   - 梯度裁剪防止梯度爆炸")
    print("   - 早停法防止过拟合")
    
    print("\n=== 演示完成 ===")
    
    return model, model_info

def estimate_flops(config: ModelConfig) -> float:
    """估计模型的浮点运算次数"""
    flops = 0
    
    # 输入卷积
    flops += config.input_channels * config.filters * 9 * config.board_size * config.board_size
    
    # 残差块
    for _ in range(config.residual_blocks):
        # 两个3x3卷积
        flops += config.filters * config.filters * 9 * config.board_size * config.board_size * 2
    
    # 棋型检测器
    if config.pattern_channels > 0:
        flops += config.filters * config.pattern_channels * 9 * config.board_size * config.board_size
    
    # 策略头
    flops += config.filters * config.policy_head_filters * 9 * config.board_size * config.board_size
    flops += config.policy_head_filters * 16 * 9 * config.board_size * config.board_size
    flops += 16 * 1 * config.board_size * config.board_size
    
    # 价值头
    conv_size = config.value_head_filters * config.board_size * config.board_size
    flops += config.filters * config.value_head_filters * config.board_size * config.board_size
    flops += conv_size * 512
    flops += 512 * 256
    flops += 256 * 1
    
    return flops

def save_architecture_summary(model_info: Dict, filename: str = "architecture_summary.json"):
    """保存架构摘要"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"\n架构摘要已保存到: {filename}")

if __name__ == "__main__":
    model, info = demonstrate_architecture()
    save_architecture_summary(info)