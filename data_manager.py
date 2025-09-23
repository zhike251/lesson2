"""
训练数据管理模块
提供数据增强、清理、过滤和分割功能

作者：Claude AI Engineer
日期：2025-09-22
"""

import os
import json
import pickle
import gzip
import numpy as np
import threading
import shutil
from typing import List, Dict, Tuple, Optional, Set, Any, Iterator
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
from pathlib import Path
import random
from collections import defaultdict, Counter

from training_data_collector import (
    TrainingDataCollector, GameRecord, GameResult, MoveData, DataAugmentation,
    BOARD_SIZE, EMPTY, BLACK, WHITE
)

class DataQuality(Enum):
    """数据质量等级"""
    HIGH = "high"      # 高质量数据
    MEDIUM = "medium"  # 中等质量数据
    LOW = "low"        # 低质量数据
    INVALID = "invalid" # 无效数据

@dataclass
class DataStats:
    """数据统计信息"""
    total_games: int = 0
    total_moves: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    avg_game_length: float = 0.0
    min_game_length: int = 0
    max_game_length: int = 0
    avg_thinking_time: float = 0.0
    unique_positions: int = 0
    duplicate_positions: int = 0
    data_size_mb: float = 0.0
    quality_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.quality_distribution is None:
            self.quality_distribution = {}

class DataManager:
    """训练数据管理器"""
    
    def __init__(self, data_dir: str = "training_data", 
                 cache_dir: str = "data_cache",
                 max_cache_size_gb: float = 10.0):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据目录
            cache_dir: 缓存目录
            max_cache_size_gb: 最大缓存大小（GB）
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_gb = max_cache_size_gb
        
        # 创建目录
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 数据索引
        self.data_index = {}  # 文件路径 -> 元数据
        self.position_hash_index = defaultdict(list)  # 位置哈希 -> 游戏ID列表
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 加载现有数据索引
        self._load_data_index()
    
    def add_data_file(self, filepath: str, metadata: Optional[Dict] = None) -> bool:
        """
        添加数据文件到管理器
        
        Args:
            filepath: 数据文件路径
            metadata: 元数据
            
        Returns:
            是否成功添加
        """
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                print(f"文件不存在: {filepath}")
                return False
            
            # 生成文件哈希
            file_hash = self._get_file_hash(file_path)
            
            # 检查是否已经存在
            if file_hash in [meta.get('file_hash') for meta in self.data_index.values()]:
                print(f"文件已存在: {filepath}")
                return False
            
            # 分析数据文件
            file_stats = self._analyze_data_file(file_path)
            if not file_stats:
                print(f"无法分析文件: {filepath}")
                return False
            
            # 添加到索引
            with self.lock:
                self.data_index[str(file_path)] = {
                    'file_hash': file_hash,
                    'file_size': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime,
                    'stats': file_stats,
                    'metadata': metadata or {},
                    'quality': self._assess_data_quality(file_stats),
                    'indexed_time': time.time()
                }
            
            # 保存索引
            self._save_data_index()
            
            print(f"成功添加数据文件: {filepath}")
            return True
            
        except Exception as e:
            print(f"添加数据文件失败: {e}")
            return False
    
    def get_data_statistics(self) -> DataStats:
        """获取数据统计信息"""
        stats = DataStats()
        
        with self.lock:
            for file_meta in self.data_index.values():
                file_stats = file_meta['stats']
                
                stats.total_games += file_stats['total_games']
                stats.total_moves += file_stats['total_moves']
                stats.black_wins += file_stats['black_wins']
                stats.white_wins += file_stats['white_wins']
                stats.draws += file_stats['draws']
                
                if file_meta['quality'] in stats.quality_distribution:
                    stats.quality_distribution[file_meta['quality']] += file_stats['total_games']
                else:
                    stats.quality_distribution[file_meta['quality']] = file_stats['total_games']
                
                stats.data_size_mb += file_meta['file_size'] / (1024 * 1024)
        
        # 计算平均值
        if stats.total_games > 0:
            stats.avg_game_length = stats.total_moves / stats.total_games
        
        return stats
    
    def clean_data(self, criteria: Dict[str, Any]) -> Dict[str, int]:
        """
        清理数据
        
        Args:
            criteria: 清理条件
                - min_game_length: 最小游戏长度
                - max_game_length: 最大游戏长度
                - min_thinking_time: 最小思考时间
                - remove_duplicates: 是否移除重复位置
                - quality_threshold: 质量阈值
                
        Returns:
            清理统计信息
        """
        print("开始数据清理...")
        
        clean_stats = {
            'processed_files': 0,
            'removed_games': 0,
            'removed_moves': 0,
            'duplicate_positions': 0,
            'cleaned_files': 0
        }
        
        min_game_length = criteria.get('min_game_length', 10)
        max_game_length = criteria.get('max_game_length', 400)
        min_thinking_time = criteria.get('min_thinking_time', 0.1)
        remove_duplicates = criteria.get('remove_duplicates', True)
        quality_threshold = criteria.get('quality_threshold', DataQuality.LOW)
        
        with self.lock:
            files_to_process = list(self.data_index.keys())
        
        for filepath in files_to_process:
            try:
                cleaned_games = self._clean_file_data(
                    filepath, min_game_length, max_game_length, 
                    min_thinking_time, remove_duplicates, quality_threshold
                )
                
                if cleaned_games['removed_games'] > 0:
                    clean_stats['cleaned_files'] += 1
                
                clean_stats['processed_files'] += 1
                clean_stats['removed_games'] += cleaned_games['removed_games']
                clean_stats['removed_moves'] += cleaned_games['removed_moves']
                clean_stats['duplicate_positions'] += cleaned_games['duplicate_positions']
                
            except Exception as e:
                print(f"清理文件失败 {filepath}: {e}")
        
        print(f"数据清理完成: {clean_stats}")
        return clean_stats
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                     test_ratio: float = 0.1, random_seed: int = 42) -> Dict[str, str]:
        """
        分割数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            
        Returns:
            分割后的文件路径
        """
        print("开始数据集分割...")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("比例之和必须等于1.0")
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 收集所有游戏数据
        all_games = []
        
        with self.lock:
            files_to_process = list(self.data_index.keys())
        
        for filepath in files_to_process:
            try:
                games = self._load_games_from_file(filepath)
                all_games.extend(games)
            except Exception as e:
                print(f"加载文件失败 {filepath}: {e}")
        
        if not all_games:
            raise ValueError("没有找到有效的游戏数据")
        
        # 随机打乱
        random.shuffle(all_games)
        
        # 分割数据
        total_games = len(all_games)
        train_end = int(total_games * train_ratio)
        val_end = train_end + int(total_games * val_ratio)
        
        train_games = all_games[:train_end]
        val_games = all_games[train_end:val_end]
        test_games = all_games[val_end:]
        
        # 保存分割后的数据
        timestamp = int(time.time())
        
        train_file = self.data_dir / f"train_split_{timestamp}.json"
        val_file = self.data_dir / f"val_split_{timestamp}.json"
        test_file = self.data_dir / f"test_split_{timestamp}.json"
        
        self._save_games_to_file(train_games, train_file)
        self._save_games_to_file(val_games, val_file)
        self._save_games_to_file(test_games, test_file)
        
        split_info = {
            'train_file': str(train_file),
            'val_file': str(val_file),
            'test_file': str(test_file),
            'train_games': len(train_games),
            'val_games': len(val_games),
            'test_games': len(test_games),
            'split_time': timestamp
        }
        
        # 保存分割信息
        split_info_file = self.data_dir / f"split_info_{timestamp}.json"
        with open(split_info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        print(f"数据集分割完成:")
        print(f"  训练集: {len(train_games)} 游戏 -> {train_file}")
        print(f"  验证集: {len(val_games)} 游戏 -> {val_file}")
        print(f"  测试集: {len(test_games)} 游戏 -> {test_file}")
        
        return split_info
    
    def augment_dataset(self, input_file: str, output_file: str, 
                       augmentation_factor: int = 8) -> Dict[str, int]:
        """
        数据增强
        
        Args:
            input_file: 输入文件
            output_file: 输出文件
            augmentation_factor: 增强倍数（最多8倍：4个旋转 + 2个镜像）
            
        Returns:
            增强统计信息
        """
        print(f"开始数据增强: {input_file} -> {output_file}")
        
        # 加载原始数据
        games = self._load_games_from_file(input_file)
        if not games:
            raise ValueError(f"无法加载游戏数据: {input_file}")
        
        augmented_games = []
        augmentation_types = ["rotate_90", "rotate_180", "rotate_270", 
                             "flip_h", "flip_v"]
        
        # 限制增强类型数量
        selected_augmentations = augmentation_types[:min(augmentation_factor - 1, len(augmentation_types))]
        
        for game in games:
            # 添加原始游戏
            augmented_games.append(game)
            
            # 添加增强版本
            for aug_type in selected_augmentations:
                try:
                    augmented_game = self._augment_game(game, aug_type)
                    augmented_games.append(augmented_game)
                except Exception as e:
                    print(f"游戏增强失败: {e}")
        
        # 保存增强后的数据
        self._save_games_to_file(augmented_games, output_file)
        
        stats = {
            'original_games': len(games),
            'augmented_games': len(augmented_games),
            'augmentation_factor': len(augmented_games) / len(games) if games else 0
        }
        
        print(f"数据增强完成: {stats}")
        return stats
    
    def merge_datasets(self, input_files: List[str], output_file: str,
                      remove_duplicates: bool = True) -> Dict[str, int]:
        """
        合并数据集
        
        Args:
            input_files: 输入文件列表
            output_file: 输出文件
            remove_duplicates: 是否移除重复数据
            
        Returns:
            合并统计信息
        """
        print(f"开始合并数据集: {len(input_files)} 个文件 -> {output_file}")
        
        all_games = []
        game_hashes = set() if remove_duplicates else None
        
        for input_file in input_files:
            try:
                games = self._load_games_from_file(input_file)
                
                for game in games:
                    if remove_duplicates:
                        game_hash = self._get_game_hash(game)
                        if game_hash in game_hashes:
                            continue
                        game_hashes.add(game_hash)
                    
                    all_games.append(game)
                    
            except Exception as e:
                print(f"加载文件失败 {input_file}: {e}")
        
        # 保存合并后的数据
        self._save_games_to_file(all_games, output_file)
        
        stats = {
            'input_files': len(input_files),
            'total_games': len(all_games),
            'duplicates_removed': len(game_hashes) - len(all_games) if remove_duplicates else 0
        }
        
        print(f"数据集合并完成: {stats}")
        return stats
    
    def export_training_samples(self, input_file: str, output_dir: str,
                               format_type: str = "numpy") -> Dict[str, str]:
        """
        导出训练样本
        
        Args:
            input_file: 输入文件
            output_dir: 输出目录
            format_type: 格式类型 ("numpy", "torch", "tensorflow")
            
        Returns:
            导出文件路径
        """
        print(f"开始导出训练样本: {input_file} -> {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 加载游戏数据
        games = self._load_games_from_file(input_file)
        if not games:
            raise ValueError(f"无法加载游戏数据: {input_file}")
        
        # 提取训练样本
        board_states = []
        move_probabilities = []
        values = []
        
        for game in games:
            for move in game.moves:
                board_states.append(move.board_state)
                
                # 转换概率分布为数组
                prob_array = np.zeros((BOARD_SIZE, BOARD_SIZE))
                for (row, col), prob in move.mcts_probabilities.items():
                    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                        prob_array[row, col] = prob
                move_probabilities.append(prob_array)
                
                values.append(move.value_estimate)
        
        # 转换为numpy数组
        board_states = np.array(board_states)
        move_probabilities = np.array(move_probabilities)
        values = np.array(values)
        
        # 保存数据
        timestamp = int(time.time())
        
        if format_type == "numpy":
            board_file = output_path / f"board_states_{timestamp}.npy"
            prob_file = output_path / f"move_probabilities_{timestamp}.npy"
            value_file = output_path / f"values_{timestamp}.npy"
            
            np.save(board_file, board_states)
            np.save(prob_file, move_probabilities)
            np.save(value_file, values)
            
            export_info = {
                'board_states': str(board_file),
                'move_probabilities': str(prob_file),
                'values': str(value_file),
                'format': 'numpy',
                'samples': len(board_states)
            }
        
        elif format_type == "torch":
            import torch
            
            torch_file = output_path / f"training_data_{timestamp}.pt"
            torch.save({
                'board_states': torch.tensor(board_states, dtype=torch.float32),
                'move_probabilities': torch.tensor(move_probabilities, dtype=torch.float32),
                'values': torch.tensor(values, dtype=torch.float32)
            }, torch_file)
            
            export_info = {
                'training_data': str(torch_file),
                'format': 'torch',
                'samples': len(board_states)
            }
        
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
        
        # 保存元数据
        metadata_file = output_path / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(export_info, f, indent=2, ensure_ascii=False)
        
        print(f"训练样本导出完成: {len(board_states)} 个样本")
        return export_info
    
    def _analyze_data_file(self, filepath: Path) -> Optional[Dict]:
        """分析数据文件"""
        try:
            if filepath.suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    games_data = data.get('games', [])
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    games_data = data.get('games', [])
            
            if not games_data:
                return None
            
            stats = {
                'total_games': len(games_data),
                'total_moves': 0,
                'black_wins': 0,
                'white_wins': 0,
                'draws': 0,
                'game_lengths': [],
                'thinking_times': []
            }
            
            for game_data in games_data:
                if isinstance(game_data, dict):
                    moves = game_data.get('moves', [])
                    stats['total_moves'] += len(moves)
                    stats['game_lengths'].append(len(moves))
                    
                    result = game_data.get('final_result', 0)
                    if result == 1:
                        stats['black_wins'] += 1
                    elif result == 2:
                        stats['white_wins'] += 1
                    else:
                        stats['draws'] += 1
                    
                    # 收集思考时间
                    for move in moves:
                        thinking_time = move.get('thinking_time', 0)
                        if thinking_time > 0:
                            stats['thinking_times'].append(thinking_time)
            
            return stats
            
        except Exception as e:
            print(f"分析文件失败: {e}")
            return None
    
    def _assess_data_quality(self, stats: Dict) -> str:
        """评估数据质量"""
        if stats['total_games'] == 0:
            return DataQuality.INVALID.value
        
        avg_game_length = stats['total_moves'] / stats['total_games']
        
        # 质量评估规则
        if avg_game_length < 10:
            return DataQuality.LOW.value
        elif avg_game_length < 50:
            return DataQuality.MEDIUM.value
        else:
            return DataQuality.HIGH.value
    
    def _get_file_hash(self, filepath: Path) -> str:
        """获取文件哈希"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_game_hash(self, game: GameRecord) -> str:
        """获取游戏哈希"""
        game_str = f"{game.game_length}_{game.final_result.value}"
        for move in game.moves[:10]:  # 只使用前10步计算哈希
            game_str += f"_{move.position[0]}_{move.position[1]}"
        return hashlib.md5(game_str.encode()).hexdigest()
    
    def _load_games_from_file(self, filepath: str) -> List[GameRecord]:
        """从文件加载游戏数据"""
        file_path = Path(filepath)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                games_data = data.get('games', [])
                return [GameRecord.from_dict(game_data) for game_data in games_data]
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data.get('games', [])
    
    def _save_games_to_file(self, games: List[GameRecord], filepath: str):
        """保存游戏数据到文件"""
        file_path = Path(filepath)
        
        games_data = [game.to_dict() for game in games]
        
        if file_path.suffix == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'games': games_data,
                    'metadata': {
                        'total_games': len(games),
                        'created_at': time.time()
                    }
                }, f, indent=2, ensure_ascii=False)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'games': games,
                    'metadata': {
                        'total_games': len(games),
                        'created_at': time.time()
                    }
                }, f)
    
    def _augment_game(self, game: GameRecord, augmentation_type: str) -> GameRecord:
        """增强单个游戏"""
        augmented_moves = []
        
        for move in game.moves:
            augmented_move = DataAugmentation.augment_move_data(move, augmentation_type)
            augmented_moves.append(augmented_move)
        
        augmented_game = GameRecord(
            game_id=f"{game.game_id}_aug_{augmentation_type}",
            moves=augmented_moves,
            final_result=game.final_result,
            game_length=game.game_length,
            total_time=game.total_time,
            black_player=game.black_player,
            white_player=game.white_player,
            metadata={**game.metadata, 'augmentation_type': augmentation_type},
            created_at=time.time()
        )
        
        return augmented_game
    
    def _clean_file_data(self, filepath: str, min_game_length: int, 
                        max_game_length: int, min_thinking_time: float,
                        remove_duplicates: bool, quality_threshold: DataQuality) -> Dict[str, int]:
        """清理文件数据"""
        games = self._load_games_from_file(filepath)
        
        cleaned_games = []
        stats = {'removed_games': 0, 'removed_moves': 0, 'duplicate_positions': 0}
        
        for game in games:
            # 检查游戏长度
            if not (min_game_length <= game.game_length <= max_game_length):
                stats['removed_games'] += 1
                continue
            
            # 清理移动数据
            cleaned_moves = []
            for move in game.moves:
                if move.thinking_time >= min_thinking_time:
                    cleaned_moves.append(move)
                else:
                    stats['removed_moves'] += 1
            
            if cleaned_moves:
                game.moves = cleaned_moves
                game.game_length = len(cleaned_moves)
                cleaned_games.append(game)
            else:
                stats['removed_games'] += 1
        
        # 保存清理后的数据
        if cleaned_games != games:
            self._save_games_to_file(cleaned_games, filepath)
        
        return stats
    
    def _load_data_index(self):
        """加载数据索引"""
        index_file = self.data_dir / "data_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.data_index = json.load(f)
            except Exception as e:
                print(f"加载数据索引失败: {e}")
                self.data_index = {}
    
    def _save_data_index(self):
        """保存数据索引"""
        index_file = self.data_dir / "data_index.json"
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.data_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存数据索引失败: {e}")

def test_data_manager():
    """测试数据管理器"""
    print("测试数据管理器...")
    
    # 创建数据管理器
    manager = DataManager("test_data")
    
    # 创建测试数据
    collector = TrainingDataCollector("test_data")
    
    # 模拟一些游戏数据
    for i in range(3):
        game_id = collector.start_game(f"Player_A_{i}", f"Player_B_{i}")
        
        # 模拟几步移动
        board = np.zeros((15, 15), dtype=int)
        board[7+i][7] = 1
        
        collector.record_move(
            position=(7+i, 7),
            player=1,
            board_state=board,
            mcts_probabilities={(7+i, 7): 0.8, (7+i, 8): 0.2},
            value_estimate=0.1 * i,
            thinking_time=1.0 + i
        )
        
        collector.end_game(GameResult.BLACK_WIN, total_time=5.0)
    
    # 保存测试数据
    test_file = collector.save_data("test_games.json", "json")
    print(f"测试数据已保存: {test_file}")
    
    # 添加数据文件到管理器
    manager.add_data_file(test_file)
    
    # 获取统计信息
    stats = manager.get_statistics()
    print(f"数据统计: {stats}")
    
    # 数据分割
    split_info = manager.split_dataset(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    print(f"数据分割: {split_info}")
    
    # 数据增强
    aug_stats = manager.augment_dataset(
        split_info['train_file'], 
        "test_data/augmented_train.json",
        augmentation_factor=4
    )
    print(f"数据增强: {aug_stats}")
    
    print("测试完成！")

if __name__ == "__main__":
    test_data_manager()