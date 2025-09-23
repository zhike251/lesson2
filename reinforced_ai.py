#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强的强化学习AI系统
集成训练后的强化学习模型

作者：Claude AI Engineer
日期：2025-09-23
"""

import time
import json
import os
import pickle
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from integrated_ai import IntegratedGomokuAI
from reinforcement_learning import SelfPlayEngine, TrainingConfig
from neural_mcts import NeuralMCTS, AlphaZeroStyleNetwork

class ReinforcedAI(IntegratedGomokuAI):
    """强化学习增强的AI系统"""
    
    def __init__(self, model_path: str = None, training_stats_path: str = None, **kwargs):
        """
        初始化强化学习AI
        
        Args:
            model_path: 训练后的模型路径
            training_stats_path: 训练统计信息路径
            **kwargs: 其他参数
        """
        # 设置强化学习特定的参数
        kwargs['ai_difficulty'] = 'reinforced'
        kwargs['use_neural'] = True
        kwargs['engine_type'] = 'neural_mcts'
        
        super().__init__(**kwargs)
        
        # 加载训练后的模型
        self.trained_network = None
        self.training_stats = None
        self.model_info = None
        
        if model_path and os.path.exists(model_path):
            self.load_trained_model(model_path)
        
        if training_stats_path and os.path.exists(training_stats_path):
            self.load_training_stats(training_stats_path)
        
        # 如果有训练后的网络，替换默认网络
        if self.trained_network:
            self.ai_engine.neural_network = self.trained_network
            self.neural_evaluator = self.trained_network.neural_evaluator
            print("✓ 已加载训练后的强化学习模型")
    
    def load_trained_model(self, model_path: str):
        """加载训练后的模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 重建神经网络
            self.trained_network = AlphaZeroStyleNetwork()
            self.trained_network.set_state(model_data['network_state'])
            
            self.model_info = model_data.get('model_info', {})
            print(f"✓ 模型加载成功: {model_path}")
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            self.trained_network = None
    
    def load_training_stats(self, stats_path: str):
        """加载训练统计信息"""
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.training_stats = data.get('training_stats', {})
            self.model_info = data.get('model_performance', {})
            print(f"✓ 训练统计加载成功: {stats_path}")
            
        except Exception as e:
            print(f"✗ 训练统计加载失败: {e}")
            self.training_stats = None
    
    def get_ai_info(self) -> Dict[str, Any]:
        """获取AI信息"""
        info = super().get_ai_info()
        
        # 添加强化学习特定信息
        if self.training_stats:
            info.update({
                'type': 'Reinforcement Learning AI',
                'training_iterations': self.training_stats.get('iteration', 0),
                'training_games': self.training_stats.get('total_games', 0),
                'evaluation_results': self.training_stats.get('evaluation_results', []),
                'is_trained': True
            })
        
        if self.model_info:
            info.update(self.model_info)
        
        return info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = super().get_performance_stats()
        
        # 添加强化学习性能统计
        if self.training_stats:
            stats['reinforcement_learning'] = {
                'training_iterations': self.training_stats.get('iteration', 0),
                'total_training_games': self.training_stats.get('total_games', 0),
                'total_training_moves': self.training_stats.get('total_moves', 0),
                'latest_evaluation': (self.training_stats.get('evaluation_results', [])[-1] 
                                    if self.training_stats.get('evaluation_results') else None)
            }
        
        return stats

class ReinforcementModelManager:
    """强化学习模型管理器"""
    
    def __init__(self, base_path: str = "rl_models"):
        """
        初始化模型管理器
        
        Args:
            base_path: 模型基础路径
        """
        self.base_path = base_path
        self.models = {}
        self.current_model = None
        
        # 创建基础目录
        os.makedirs(base_path, exist_ok=True)
        
        # 扫描已有模型
        self._scan_models()
    
    def _scan_models(self):
        """扫描已有模型"""
        if not os.path.exists(self.base_path):
            return
        
        for filename in os.listdir(self.base_path):
            if filename.endswith('.pkl') and filename.startswith('model_'):
                model_path = os.path.join(self.base_path, filename)
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    iteration = model_data.get('iteration', 'unknown')
                    self.models[iteration] = {
                        'path': model_path,
                        'data': model_data,
                        'info': model_data.get('model_info', {})
                    }
                    
                    print(f"发现模型: iteration {iteration}")
                    
                except Exception as e:
                    print(f"扫描模型失败 {filename}: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用模型"""
        model_list = []
        
        for iteration, model_info in self.models.items():
            info = {
                'iteration': iteration,
                'path': model_info['path'],
                'stats': model_info['data'].get('training_stats', {}),
                'performance': model_info['info']
            }
            model_list.append(info)
        
        return sorted(model_list, key=lambda x: x['iteration'])
    
    def load_model(self, iteration: str = 'latest') -> Optional[ReinforcedAI]:
        """加载指定模型"""
        if iteration == 'latest':
            # 获取最新的模型
            available_iterations = [int(i) for i in self.models.keys() if i.isdigit()]
            if not available_iterations:
                print("没有可用的训练模型")
                return None
            
            iteration = str(max(available_iterations))
        
        if iteration not in self.models:
            print(f"模型 {iteration} 不存在")
            return None
        
        model_info = self.models[iteration]
        model_path = model_info['path']
        
        # 查找对应的训练统计文件
        stats_path = os.path.join(self.base_path, f"training_data_{iteration}.json")
        if not os.path.exists(stats_path):
            stats_path = None
        
        # 创建强化学习AI
        ai = ReinforcedAI(
            model_path=model_path,
            training_stats_path=stats_path
        )
        
        self.current_model = ai
        print(f"✓ 已加载模型 {iteration}")
        
        return ai
    
    def get_best_model(self) -> Optional[ReinforcedAI]:
        """获取最佳模型（基于评估结果）"""
        best_iteration = None
        best_win_rate = -1
        
        for iteration, model_info in self.models.items():
            stats = model_info['data'].get('training_stats', {})
            evaluation_results = stats.get('evaluation_results', [])
            
            if evaluation_results:
                latest_win_rate = evaluation_results[-1]
                if latest_win_rate > best_win_rate:
                    best_win_rate = latest_win_rate
                    best_iteration = iteration
        
        if best_iteration:
            return self.load_model(best_iteration)
        else:
            return self.load_model('latest')
    
    def compare_models(self, iterations: List[str] = None) -> Dict[str, Dict]:
        """比较多个模型的性能"""
        if iterations is None:
            iterations = list(self.models.keys())
        
        comparison = {}
        
        for iteration in iterations:
            if iteration in self.models:
                model_info = self.models[iteration]
                stats = model_info['data'].get('training_stats', {})
                
                comparison[iteration] = {
                    'iteration': iteration,
                    'total_games': stats.get('total_games', 0),
                    'total_moves': stats.get('total_moves', 0),
                    'evaluation_results': stats.get('evaluation_results', []),
                    'latest_win_rate': (stats.get('evaluation_results', [])[-1] 
                                      if stats.get('evaluation_results') else None)
                }
        
        return comparison
    
    def delete_model(self, iteration: str):
        """删除指定模型"""
        if iteration not in self.models:
            print(f"模型 {iteration} 不存在")
            return False
        
        model_info = self.models[iteration]
        
        # 删除模型文件
        try:
            os.remove(model_info['path'])
            print(f"✓ 已删除模型文件: {model_info['path']}")
        except Exception as e:
            print(f"✗ 删除模型文件失败: {e}")
        
        # 删除训练数据文件
        data_path = os.path.join(self.base_path, f"training_data_{iteration}.json")
        if os.path.exists(data_path):
            try:
                os.remove(data_path)
                print(f"✓ 已删除训练数据: {data_path}")
            except Exception as e:
                print(f"✗ 删除训练数据失败: {e}")
        
        # 从内存中移除
        del self.models[iteration]
        
        if self.current_model and self.current_model.model_info.get('iteration') == iteration:
            self.current_model = None
        
        print(f"✓ 模型 {iteration} 已删除")
        return True

def create_reinforced_ai_from_latest_training() -> Optional[ReinforcedAI]:
    """从最新训练创建强化学习AI"""
    manager = ReinforcementModelManager()
    return manager.load_model('latest')

def create_reinforced_ai_from_best_model() -> Optional[ReinforcedAI]:
    """从最佳模型创建强化学习AI"""
    manager = ReinforcementModelManager()
    return manager.get_best_model()

def get_reinforcement_training_summary() -> Optional[Dict[str, Any]]:
    """获取强化学习训练总结"""
    summary_path = "rl_training_summary.json"
    
    if not os.path.exists(summary_path):
        return None
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取训练总结失败: {e}")
        return None

if __name__ == "__main__":
    """测试强化学习AI系统"""
    print("强化学习AI系统测试")
    print("=" * 40)
    
    # 1. 创建模型管理器
    manager = ReinforcementModelManager()
    
    # 2. 列出可用模型
    models = manager.list_models()
    print(f"发现 {len(models)} 个模型:")
    
    for model in models:
        iteration = model['iteration']
        stats = model['stats']
        latest_win_rate = model['performance'].get('evaluation_results', [])[-1] if model['performance'].get('evaluation_results') else None
        
        print(f"  模型 {iteration}:")
        print(f"    训练游戏: {stats.get('total_games', 0)}")
        print(f"    训练移动: {stats.get('total_moves', 0)}")
        if latest_win_rate is not None:
            print(f"    最新胜率: {latest_win_rate:.2%}")
        print()
    
    # 3. 加载最佳模型
    if models:
        print("加载最佳模型...")
        best_ai = manager.get_best_model()
        
        if best_ai:
            info = best_ai.get_ai_info()
            print(f"✓ 成功加载强化学习AI")
            print(f"  类型: {info.get('type', 'Unknown')}")
            print(f"  训练迭代: {info.get('training_iterations', 0)}")
            print(f"  训练游戏: {info.get('training_games', 0)}")
            
            if info.get('evaluation_results'):
                print(f"  评估胜率: {info['evaluation_results'][-1]:.2%}")
        else:
            print("✗ 加载模型失败")
    else:
        print("没有可用的训练模型")
        print("请先运行训练脚本: python train_reinforcement_learning.py")