#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
强化学习训练脚本
简化版本，专注于快速训练和测试

作者：Claude AI Engineer
日期：2025-09-23
"""

import time
import numpy as np
import json
import os
import pickle
from typing import List, Dict, Tuple
import random

from reinforcement_learning import ReinforcementLearningTrainer, TrainingConfig
from neural_mcts import NeuralMCTS, AlphaZeroStyleNetwork
from neural_evaluator import NeuralNetworkEvaluator, NeuralConfig

def create_training_config() -> TrainingConfig:
    """创建训练配置"""
    return TrainingConfig(
        # 训练参数（快速测试配置）
        num_iterations=20,  # 20次迭代
        games_per_iteration=5,  # 每次迭代5局游戏
        batch_size=16,  # 小批次
        epochs_per_iteration=3,  # 每次迭代3轮训练
        
        # MCTS参数
        mcts_simulations=100,  # 减少模拟次数以加快速度
        c_puct=1.25,
        temperature=1.0,
        temperature_drop=0.5,
        
        # 神经网络参数
        learning_rate=0.001,
        weight_decay=0.0001,
        dropout_rate=0.3,
        
        # 经验回放
        replay_buffer_size=1000,  # 较小的缓冲区
        replay_buffer_sample_size=500,
        
        # 保存和评估
        save_interval=5,  # 每5次迭代保存
        evaluation_interval=3,  # 每3次迭代评估
        evaluation_games=20,  # 评估游戏数
        
        # 路径配置
        model_save_path="rl_models",
        data_save_path="rl_data",
        log_path="rl_logs"
    )

def run_quick_training():
    """运行快速训练"""
    print("=== 强化学习快速训练 ===")
    
    # 创建配置
    config = create_training_config()
    
    # 显示配置信息
    print(f"训练配置:")
    print(f"  迭代次数: {config.num_iterations}")
    print(f"  每迭代游戏数: {config.games_per_iteration}")
    print(f"  MCTS模拟次数: {config.mcts_simulations}")
    print(f"  批次大小: {config.batch_size}")
    
    # 创建训练器
    trainer = ReinforcementLearningTrainer(config)
    
    # 开始训练
    start_time = time.time()
    trained_engine = trainer.train()
    end_time = time.time()
    
    # 训练完成
    training_time = end_time - start_time
    print(f"\n=== 训练完成 ===")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"平均每迭代时间: {training_time/config.num_iterations:.2f} 秒")
    print(f"总游戏数: {config.num_iterations * config.games_per_iteration}")
    
    # 显示训练统计
    stats = trained_engine.training_stats
    print(f"\n训练统计:")
    print(f"  总移动数: {stats['total_moves']}")
    print(f"  评估结果: {len(stats['evaluation_results'])} 次")
    if stats['evaluation_results']:
        print(f"  最新胜率: {stats['evaluation_results'][-1]:.2%}")
    
    return trained_engine

def test_trained_model(trained_engine, num_games=10):
    """测试训练后的模型"""
    print(f"\n=== 测试训练后的模型 ===")
    print(f"进行 {num_games} 局测试游戏...")
    
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(num_games):
        if i % 3 == 0:
            print(f"  测试进度: {i+1}/{num_games}")
        
        # 进行评估游戏
        result = trained_engine.play_evaluation_game()
        
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    win_rate = wins / num_games
    print(f"\n测试结果:")
    print(f"  胜利: {wins} ({wins/num_games:.1%})")
    print(f"  失败: {losses} ({losses/num_games:.1%})")
    print(f"  平局: {draws} ({draws/num_games:.1%})")
    print(f"  胜率: {win_rate:.1%}")
    
    return win_rate

def create_enhanced_ai_with_trained_model(trained_engine):
    """创建使用训练模型的增强AI"""
    print(f"\n=== 创建增强AI ===")
    
    # 获取训练后的神经网络
    trained_network = trained_engine.neural_network
    
    # 创建增强的AI配置
    ai_config = {
        'neural_network': trained_network,
        'training_stats': trained_engine.training_stats,
        'model_info': {
            'trained': True,
            'iterations': trained_engine.training_stats['iteration'],
            'total_games': trained_engine.training_stats['total_games'],
            'evaluation_results': trained_engine.training_stats['evaluation_results']
        }
    }
    
    print("增强AI配置完成!")
    print(f"模型信息:")
    print(f"  训练迭代: {ai_config['model_info']['iterations']}")
    print(f"  训练游戏: {ai_config['model_info']['total_games']}")
    if ai_config['model_info']['evaluation_results']:
        print(f"  评估胜率: {ai_config['model_info']['evaluation_results'][-1]:.2%}")
    
    return ai_config

def save_training_summary(trained_engine, win_rate):
    """保存训练总结"""
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_stats': trained_engine.training_stats,
        'test_win_rate': win_rate,
        'model_performance': {
            'is_trained': True,
            'total_training_games': trained_engine.training_stats['total_games'],
            'final_iteration': trained_engine.training_stats['iteration'],
            'evaluation_history': trained_engine.training_stats['evaluation_results']
        }
    }
    
    # 保存总结
    summary_path = "rl_training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练总结已保存到: {summary_path}")
    return summary_path

def main():
    """主函数"""
    print("强化学习训练脚本启动")
    print("=" * 50)
    
    try:
        # 1. 运行训练
        trained_engine = run_quick_training()
        
        # 2. 测试模型
        win_rate = test_trained_model(trained_engine, num_games=10)
        
        # 3. 创建增强AI配置
        enhanced_ai_config = create_enhanced_ai_with_trained_model(trained_engine)
        
        # 4. 保存训练总结
        summary_path = save_training_summary(trained_engine, win_rate)
        
        print(f"\n=== 全部完成 ===")
        print(f"✓ 强化学习训练完成")
        print(f"✓ 模型测试完成")
        print(f"✓ 增强AI配置完成")
        print(f"✓ 训练总结已保存")
        
        print(f"\n使用方法:")
        print(f"1. 训练后的模型保存在: rl_models/")
        print(f"2. 训练数据保存在: rl_data/")
        print(f"3. 训练总结在: {summary_path}")
        print(f"4. 在游戏中使用 'reinforced' 难度来体验训练后的AI")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()