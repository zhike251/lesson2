"""
五子棋AI深度学习训练系统

提供完整的训练管道，包括：
1. 自对弈数据生成
2. 经验回放池管理
3. 课程学习策略
4. 模型评估与选择
5. 分布式训练支持

作者：Claude AI Engineer  
日期：2025-09-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import random
import json
import pickle
import time
import os
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from deep_learning_architecture import GomokuNetwork, ModelConfig, AdvancedLoss, DataAugmentation

# 训练样本数据结构
TrainingSample = namedtuple('TrainingSample', [
    'state', 'player', 'policy', 'value', 'move_history'
])

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础设置
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    
    # 优化器设置
    optimizer_type: str = "adamw"  # adam, adamw, sgd
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # 训练策略
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    accumulation_steps: int = 1
    
    # 数据设置
    replay_buffer_size: int = 100000
    self_play_games: int = 1000
    evaluation_games: int = 100
    
    # 保存设置
    save_frequency: int = 10
    evaluation_frequency: int = 5
    keep_best_models: int = 5
    
    # 分布式设置
    world_size: int = 1
    rank: int = 0
    
    # 课程学习
    curriculum_learning: bool = True
    difficulty_progression: List[str] = None
    
    def __post_init__(self):
        if self.difficulty_progression is None:
            self.difficulty_progression = ["easy", "medium", "hard", "expert"]

class ExperienceReplayBuffer:
    """经验回放池"""
    
    def __init__(self, max_size: int = 100000, min_games_before_training: int = 100):
        self.max_size = max_size
        self.min_games_before_training = min_games_before_training
        self.buffer = deque(maxlen=max_size)
        self.game_count = 0
        
    def add_game(self, game_samples: List[TrainingSample]):
        """添加一局游戏的样本"""
        for sample in game_samples:
            self.buffer.append(sample)
        self.game_count += 1
        
        # 保持缓冲区大小
        if len(self.buffer) > self.max_size:
            # 移除最旧的样本
            for _ in range(len(self.buffer) - self.max_size):
                self.buffer.popleft()
    
    def sample_batch(self, batch_size: int) -> List[TrainingSample]:
        """采样批次数据"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        return random.sample(list(self.buffer), batch_size)
    
    def ready_for_training(self) -> bool:
        """检查是否准备好开始训练"""
        return self.game_count >= self.min_games_before_training
    
    def __len__(self):
        return len(self.buffer)

class GomokuDataset(Dataset):
    """五子棋数据集"""
    
    def __init__(self, samples: List[TrainingSample], config: ModelConfig):
        self.samples = samples
        self.config = config
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 准备输入特征
        state_tensor = self._prepare_state(sample.state, sample.player, sample.move_history)
        
        # 准备策略目标
        policy_tensor = self._prepare_policy(sample.policy)
        
        # 准备价值目标
        value_tensor = torch.tensor([sample.value], dtype=torch.float32)
        
        return {
            'state': state_tensor,
            'policy': policy_tensor,
            'value': value_tensor
        }
    
    def _prepare_state(self, state: np.ndarray, player: int, 
                      move_history: List[Tuple[int, int]]) -> torch.Tensor:
        """准备状态输入"""
        # 使用与网络相同的输入准备逻辑
        batch_size = 1
        channels = self.config.input_channels
        height, width = self.config.board_size, self.config.board_size
        
        input_tensor = torch.zeros(channels, height, width)
        
        # 基础特征
        input_tensor[0] = torch.from_numpy((state == player).astype(np.float32))
        input_tensor[1] = torch.from_numpy((state == (3 - player)).astype(np.float32))
        input_tensor[2] = torch.from_numpy((state == 0).astype(np.float32))
        input_tensor[3] = float(player == 1)
        
        # 位置编码
        for i in range(height):
            for j in range(width):
                input_tensor[4, i, j] = i / (height - 1)
                input_tensor[5, i, j] = j / (width - 1)
        
        # 中心距离
        center = height // 2
        for i in range(height):
            for j in range(width):
                distance = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                input_tensor[6, i, j] = 1.0 - min(distance / center, 1.0)
        
        # 移动历史
        if move_history and len(move_history) > 0:
            for idx, step in enumerate(range(-1, -6, -1)):
                if abs(step) <= len(move_history) and 7 + idx < channels:
                    row, col = move_history[step]
                    input_tensor[7 + idx, row, col] = 1.0
        
        return input_tensor
    
    def _prepare_policy(self, policy_dict: Dict[Tuple[int, int], float]) -> torch.Tensor:
        """准备策略目标"""
        policy_tensor = torch.zeros(self.config.board_size * self.config.board_size)
        
        for (row, col), prob in policy_dict.items():
            idx = row * self.config.board_size + col
            policy_tensor[idx] = prob
        
        return policy_tensor

class SelfPlayWorker:
    """自对弈工作器"""
    
    def __init__(self, model: GomokuNetwork, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # 从现有MCTS系统导入
        try:
            from neural_mcts import NeuralMCTS
            self.mcts = NeuralMCTS(model, simulations=800)
        except ImportError:
            # 简化的MCTS实现
            self.mcts = None
            logging.warning("MCTS模块未找到，使用简化自对弈")
    
    def play_game(self) -> List[TrainingSample]:
        """进行一局自对弈"""
        samples = []
        
        # 初始化游戏
        state = np.zeros((15, 15), dtype=int)
        current_player = 1
        move_history = []
        
        game_samples = []
        
        while True:
            # 获取当前状态的策略
            if self.mcts:
                policy_dict, _ = self.mcts.search(state, current_player)
            else:
                policy_dict, _ = self.model.predict(state, current_player, move_history)
            
            # 记录训练样本
            sample = TrainingSample(
                state=state.copy(),
                player=current_player,
                policy=policy_dict.copy(),
                value=0.0,  # 将在游戏结束后回填
                move_history=move_history.copy()
            )
            game_samples.append(sample)
            
            # 根据策略采样动作
            actions = list(policy_dict.keys())
            probs = list(policy_dict.values())
            
            if not actions:
                break
                
            action = np.random.choice(len(actions), p=probs)
            row, col = actions[action]
            
            # 执行动作
            state[row, col] = current_player
            move_history.append((row, col))
            
            # 检查游戏结束
            winner = self._check_winner(state, row, col)
            if winner != 0:
                # 回填价值
                for i, sample in enumerate(game_samples):
                    if sample.player == winner:
                        game_samples[i] = sample._replace(value=1.0)
                    else:
                        game_samples[i] = sample._replace(value=-1.0)
                break
            
            # 检查平局
            if np.sum(state == 0) == 0:
                # 平局，所有样本价值为0
                for i, sample in enumerate(game_samples):
                    game_samples[i] = sample._replace(value=0.0)
                break
            
            # 切换玩家
            current_player = 3 - current_player
        
        return game_samples
    
    def _check_winner(self, state: np.ndarray, row: int, col: int) -> int:
        """检查游戏是否结束"""
        player = state[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while (0 <= x < 15 and 0 <= y < 15 and state[x, y] == player):
                count += 1
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while (0 <= x < 15 and 0 <= y < 15 and state[x, y] == player):
                count += 1
                x -= dx
                y -= dy
            
            if count >= 5:
                return player
        
        return 0

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def evaluate_model(self, model1: GomokuNetwork, model2: GomokuNetwork, 
                      num_games: int = 100) -> Dict[str, float]:
        """评估两个模型的对战结果"""
        model1.eval()
        model2.eval()
        
        wins = {'model1': 0, 'model2': 0, 'draw': 0}
        
        for game_idx in range(num_games):
            # 随机决定谁先手
            if game_idx % 2 == 0:
                result = self._play_evaluation_game(model1, model2)
                if result == 1:
                    wins['model1'] += 1
                elif result == 2:
                    wins['model2'] += 1
                else:
                    wins['draw'] += 1
            else:
                result = self._play_evaluation_game(model2, model1)
                if result == 1:
                    wins['model2'] += 1
                elif result == 2:
                    wins['model1'] += 1
                else:
                    wins['draw'] += 1
        
        # 计算胜率
        total_games = wins['model1'] + wins['model2'] + wins['draw']
        win_rate = wins['model1'] / total_games if total_games > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'wins': wins['model1'],
            'losses': wins['model2'],
            'draws': wins['draw'],
            'total_games': total_games
        }
    
    def _play_evaluation_game(self, model1: GomokuNetwork, model2: GomokuNetwork) -> int:
        """进行一局评估游戏"""
        state = np.zeros((15, 15), dtype=int)
        current_player = 1
        move_history = []
        models = {1: model1, 2: model2}
        
        while True:
            current_model = models[current_player]
            
            # 获取最佳动作
            policy_dict, _ = current_model.predict(state, current_player, move_history)
            
            if not policy_dict:
                return 0  # 平局
            
            # 选择概率最高的动作
            best_action = max(policy_dict.items(), key=lambda x: x[1])[0]
            row, col = best_action
            
            # 执行动作
            state[row, col] = current_player
            move_history.append((row, col))
            
            # 检查胜利
            if self._check_winner(state, row, col):
                return current_player
            
            # 检查平局
            if np.sum(state == 0) == 0:
                return 0
            
            # 切换玩家
            current_player = 3 - current_player
    
    def _check_winner(self, state: np.ndarray, row: int, col: int) -> bool:
        """检查是否获胜"""
        player = state[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while (0 <= x < 15 and 0 <= y < 15 and state[x, y] == player):
                count += 1
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while (0 <= x < 15 and 0 <= y < 15 and state[x, y] == player):
                count += 1
                x -= dx
                y -= dy
            
            if count >= 5:
                return True
        
        return False

class GomokuTrainer:
    """五子棋AI训练器"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        # 初始化模型
        self.model = GomokuNetwork(model_config)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 初始化损失函数
        self.loss_fn = AdvancedLoss(model_config)
        
        # 初始化数据增强
        self.data_augmentation = DataAugmentation(model_config)
        
        # 初始化经验回放池
        self.replay_buffer = ExperienceReplayBuffer(
            max_size=training_config.replay_buffer_size
        )
        
        # 初始化工作器
        self.self_play_worker = SelfPlayWorker(self.model, training_config)
        self.evaluator = ModelEvaluator(training_config)
        
        # 训练统计
        self.training_stats = {
            'epoch': 0,
            'total_games': 0,
            'best_model_path': None,
            'best_win_rate': 0.0,
            'loss_history': [],
            'evaluation_history': []
        }
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 混合精度训练
        if training_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 设置日志
        self._setup_logging()
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.training_config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                momentum=0.9,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"未知的优化器类型: {self.training_config.optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.training_config.scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.epochs
            )
        elif self.training_config.scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.epochs // 3,
                gamma=0.1
            )
        elif self.training_config.scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return None
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_self_play_data(self, num_games: int):
        """生成自对弈数据"""
        self.logger.info(f"开始生成 {num_games} 局自对弈数据...")
        
        for game_idx in range(num_games):
            # 进行自对弈
            game_samples = self.self_play_worker.play_game()
            
            # 添加到经验回放池
            self.replay_buffer.add_game(game_samples)
            
            if (game_idx + 1) % 100 == 0:
                self.logger.info(f"已完成 {game_idx + 1}/{num_games} 局自对弈")
        
        self.training_stats['total_games'] += num_games
        self.logger.info(f"自对弈数据生成完成，当前缓冲区大小: {len(self.replay_buffer)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        if not self.replay_buffer.ready_for_training():
            self.logger.warning("经验回放池数据不足，跳过训练")
            return {}
        
        self.model.train()
        
        # 采样训练数据
        training_samples = self.replay_buffer.sample_batch(
            self.training_config.batch_size * 10  # 采样更多数据用于一个epoch
        )
        
        # 创建数据集和数据加载器
        dataset = GomokuDataset(training_samples, self.model_config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 移动到设备
            states = batch['state'].to(self.device)
            policies = batch['policy'].to(self.device)
            values = batch['value'].to(self.device)
            
            # 数据增强
            if self.training_config.batch_size > 1:
                states, policies, values = self.data_augmentation.augment_batch(
                    states, policies, values
                )
            
            # 前向传播
            if self.scaler:
                with torch.cuda.amp.autocast():
                    policy_pred, value_pred = self.model(states)
                    losses = self.loss_fn(policy_pred, value_pred, policies, values)
            else:
                policy_pred, value_pred = self.model(states)
                losses = self.loss_fn(policy_pred, value_pred, policies, values)
            
            # 反向传播
            loss = losses['total_loss'] / self.training_config.accumulation_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.training_config.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 统计
            total_loss += losses['total_loss'].item()
            total_policy_loss += losses['policy_loss'].item()
            total_value_loss += losses['value_loss'].item()
            num_batches += 1
        
        # 返回平均损失
        epoch_stats = {
            'total_loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return epoch_stats
    
    def evaluate_model(self) -> Dict[str, float]:
        """评估当前模型"""
        self.logger.info("开始模型评估...")
        
        # 创建基准模型（随机策略或之前的最佳模型）
        baseline_model = GomokuNetwork(self.model_config).to(self.device)
        
        # 评估
        eval_results = self.evaluator.evaluate_model(
            self.model, baseline_model, self.training_config.evaluation_games
        )
        
        self.logger.info(f"评估结果: {eval_results}")
        return eval_results
    
    def save_model(self, filepath: str, is_best: bool = False):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_stats': self.training_stats,
            'model_config': asdict(self.model_config),
            'training_config': asdict(self.training_config)
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            self.training_stats['best_model_path'] = filepath
            self.logger.info(f"保存最佳模型: {filepath}")
        else:
            self.logger.info(f"保存检查点: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        self.logger.info(f"加载模型: {filepath}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练五子棋AI模型...")
        
        # 创建保存目录
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        
        best_win_rate = 0.0
        
        for epoch in range(self.training_config.epochs):
            epoch_start_time = time.time()
            
            # 生成自对弈数据
            if epoch % 5 == 0:  # 每5个epoch生成新数据
                self.generate_self_play_data(self.training_config.self_play_games)
            
            # 训练一个epoch
            train_stats = self.train_epoch()
            
            if train_stats:
                self.training_stats['loss_history'].append(train_stats)
                
                # 学习率调度
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(train_stats['total_loss'])
                    else:
                        self.scheduler.step()
                
                # 评估模型
                if (epoch + 1) % self.training_config.evaluation_frequency == 0:
                    eval_stats = self.evaluate_model()
                    self.training_stats['evaluation_history'].append(eval_stats)
                    
                    # 保存最佳模型
                    current_win_rate = eval_stats.get('win_rate', 0.0)
                    if current_win_rate > best_win_rate:
                        best_win_rate = current_win_rate
                        self.training_stats['best_win_rate'] = best_win_rate
                        
                        best_model_path = save_dir / f"best_model_epoch_{epoch+1}.pth"
                        self.save_model(str(best_model_path), is_best=True)
                
                # 定期保存检查点
                if (epoch + 1) % self.training_config.save_frequency == 0:
                    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
                    self.save_model(str(checkpoint_path))
                
                # 打印训练信息
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch+1}/{self.training_config.epochs} - "
                    f"Loss: {train_stats['total_loss']:.4f} - "
                    f"Policy Loss: {train_stats['policy_loss']:.4f} - "
                    f"Value Loss: {train_stats['value_loss']:.4f} - "
                    f"LR: {train_stats['learning_rate']:.6f} - "
                    f"Time: {epoch_time:.2f}s"
                )
            
            self.training_stats['epoch'] = epoch + 1
        
        self.logger.info("训练完成！")
        
        # 保存最终模型
        final_model_path = save_dir / "final_model.pth"
        self.save_model(str(final_model_path))
        
        return self.training_stats

# 使用示例
def main():
    """主函数"""
    # 模型配置
    model_config = ModelConfig(
        board_size=15,
        input_channels=12,
        residual_blocks=8,
        filters=128,
        use_attention=True,
        use_multiscale=True,
        pattern_channels=16
    )
    
    # 训练配置
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=100,
        self_play_games=500,
        evaluation_games=50
    )
    
    # 创建训练器
    trainer = GomokuTrainer(model_config, training_config)
    
    # 开始训练
    training_stats = trainer.train()
    
    print("训练统计:")
    print(json.dumps(training_stats, indent=2, default=str))

if __name__ == "__main__":
    main()