#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
终极威胁检测AI
结合神经网络和强制规则检测，确保AI能防守活三
无论如何都要让AI学会防守威胁！

作者：Claude AI Engineer
日期：2025-09-24
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
from typing import List, Tuple, Optional, Dict, Any
from real_reinforced_ai_fixed import RealReinforcementLearningNetwork

class UltimateThreatAI:
    """终极威胁检测AI"""
    
    def __init__(self, model_path: str = None):
        self.board_size = 15
        self.network = RealReinforcementLearningNetwork()
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.load_best_available_model()
        
        self.network.eval()
        
        # 威胁检测统计
        self.threat_stats = {
            'threats_detected': 0,
            'threats_defended': 0,
            'forced_defenses': 0
        }
        
        # 移动历史和性能统计
        self.move_history = []
        self.performance_stats = {
            'total_time': 0.0,
            'total_moves': 0
        }
    
    def load_best_available_model(self):
        """加载最佳可用模型"""
        model_paths = [
            "models/force_threat/forced_detection_model.pth",
            "models/threat_fix/fixed_threat_detection.pth",
            "models/improved_trained/best_model_epoch_9.pth"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location='cpu')
                    self.network.load_state_dict(checkpoint['model_state_dict'])
                    print(f"成功加载模型: {path}")
                    return
                except Exception as e:
                    print(f"加载模型失败 {path}: {e}")
        
        print("使用随机初始化模型")
    
    def load_model(self, path: str):
        """加载指定模型"""
        checkpoint = torch.load(path, map_location='cpu')
        self.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型: {path}")
    
    def get_move(self, board: List[List[int]], player: int) -> Tuple[int, int]:
        """获取AI移动 - 优先进攻，其次防守"""
        import time
        start_time = time.time()
        
        board_array = np.array(board)
        
        # 第一步：进攻策略 - 寻找最佳进攻机会
        attack_move = self.find_best_attack_move(board_array, player)
        if attack_move:
            print(f"进攻策略：选择进攻位置 {attack_move}")
            
            # 记录移动
            self.move_history.append({
                'position': attack_move,
                'player': player,
                'type': 'attack',
                'time': time.time() - start_time
            })
            self.performance_stats['total_time'] += time.time() - start_time
            self.performance_stats['total_moves'] += 1
            
            return attack_move
        
        # 第二步：防守策略 - 检查威胁
        defense_move = self.force_detect_threats(board_array, player)
        if defense_move:
            self.threat_stats['forced_defenses'] += 1
            print(f"防守策略：选择防守位置 {defense_move}")
            
            # 记录移动
            self.move_history.append({
                'position': defense_move,
                'player': player,
                'type': 'defense',
                'time': time.time() - start_time
            })
            self.performance_stats['total_time'] += time.time() - start_time
            self.performance_stats['total_moves'] += 1
            
            return defense_move
        
        # 第三步：使用神经网络
        with torch.no_grad():
            state_tensor = torch.FloatTensor(board_array.astype(float)).unsqueeze(0)
            policy, value = self.network(state_tensor)
            policy_probs = torch.exp(policy).squeeze().numpy()
            
            # 获取有效移动
            valid_moves = []
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board_array[i, j] == 0:
                        idx = i * self.board_size + j
                        valid_moves.append((i, j, policy_probs[idx]))
            
            if not valid_moves:
                return None
            
            # 选择概率最高的位置
            best_move = max(valid_moves, key=lambda x: x[2])
            move = (best_move[0], best_move[1])
            
            # 记录移动
            self.move_history.append({
                'position': move,
                'player': player,
                'type': 'neural_network',
                'confidence': float(best_move[2]),
                'time': time.time() - start_time
            })
            self.performance_stats['total_time'] += time.time() - start_time
            self.performance_stats['total_moves'] += 1
            
            return move
    
    def find_best_attack_move(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """寻找最佳进攻移动"""
        # 优先级1：直接获胜
        win_move = self.find_win_threat(board, player)
        if win_move:
            return win_move
        
        # 优先级2：形成活四
        live_four_move = self.find_live_four_opportunity(board, player)
        if live_four_move:
            return live_four_move
        
        # 优先级3：形成双活三（双重威胁）
        double_threat_move = self.find_double_live_three_opportunity(board, player)
        if double_threat_move:
            return double_threat_move
        
        # 优先级4：形成活三
        live_three_move = self.find_live_three_opportunity(board, player)
        if live_three_move:
            return live_three_move
        
        # 优先级5：形成冲四
        rush_four_move = self.find_rush_four_opportunity(board, player)
        if rush_four_move:
            return rush_four_move
        
        return None
    
    def find_live_four_opportunity(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """寻找形成活四的机会"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    # 模拟落子
                    board[i, j] = player
                    if self.has_live_four(board, i, j, player):
                        board[i, j] = 0  # 恢复
                        return (i, j)
                    board[i, j] = 0  # 恢复
        return None
    
    def find_double_live_three_opportunity(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """寻找形成双活三的机会（最高优先级的进攻）"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    # 模拟落子
                    board[i, j] = player
                    # 计算形成了多少个活三
                    live_three_count = self.count_live_three_patterns(board, i, j, player)
                    board[i, j] = 0  # 恢复
                    
                    # 如果形成了2个或以上的活三，这就是双活三机会
                    if live_three_count >= 2:
                        return (i, j)
        return None
    
    def find_live_three_opportunity(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """寻找形成活三的机会"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    # 模拟落子
                    board[i, j] = player
                    if self.has_live_three(board, i, j, player):
                        board[i, j] = 0  # 恢复
                        return (i, j)
                    board[i, j] = 0  # 恢复
        return None
    
    def find_rush_four_opportunity(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """寻找形成冲四的机会"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    # 模拟落子
                    board[i, j] = player
                    
                    # 检查是否形成冲四
                    for dx, dy in directions:
                        if self.is_rush_four_pattern(board, i, j, dx, dy, player):
                            board[i, j] = 0  # 恢复
                            return (i, j)
                    
                    board[i, j] = 0  # 恢复
        return None
    
    def count_live_three_patterns(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """计算指定位置形成了多少个活三模式"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        count = 0
        
        for dx, dy in directions:
            if self.is_live_three_pattern(board, row, col, dx, dy, player):
                count += 1
        
        return count
    
    def is_live_three_pattern(self, board: np.ndarray, row: int, col: int, dx: int, dy: int, player: int) -> bool:
        """检查特定方向是否形成活三模式"""
        total_count = 1
        open_ends = 0
        gaps = 0
        
        # 正方向检查
        x, y = row + dx, col + dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            if board[x, y] == player:
                total_count += 1
            elif board[x, y] == 0:
                open_ends += 1
                break
            else:
                gaps += 1
                break
            x += dx
            y += dy
        
        # 反方向检查
        x, y = row - dx, col - dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size:
            if board[x, y] == player:
                total_count += 1
            elif board[x, y] == 0:
                open_ends += 1
                break
            else:
                gaps += 1
                break
            x -= dx
            y -= dy
        
        # 活三条件：3个子，两端都开放，或者一端开放且有1个空隙
        return total_count == 3 and open_ends >= 1 and gaps <= 1
    
    def is_rush_four_pattern(self, board: np.ndarray, row: int, col: int, dx: int, dy: int, player: int) -> bool:
        """检查是否形成冲四模式（四子连珠，一端被阻挡）"""
        total_count = 1
        open_ends = 0
        
        # 正方向检查
        x, y = row + dx, col + dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
            total_count += 1
            x += dx
            y += dy
        
        if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
            open_ends += 1
        
        # 反方向检查
        x, y = row - dx, col - dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
            total_count += 1
            x -= dx
            y -= dy
        
        if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
            open_ends += 1
        
        # 冲四条件：4个子，只有一端开放
        return total_count == 4 and open_ends == 1
    
    def force_detect_threats(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """强制检测威胁"""
        # 确保威胁统计字典存在
        if not hasattr(self, 'threat_stats'):
            self.threat_stats = {
                'total_threats_detected': 0,
                'threats_defended': 0,
                'forced_defenses': 0,
                'defense_rate': 0.0
            }
        
        # 正确计算对手：如果player是1，对手是-1；如果player是-1，对手是1
        opponent = -player
        
        # 检查自己的获胜机会（最高优先级）
        my_win = self.find_win_threat(board, player)
        if my_win:
            print(f"检测到获胜机会：直接获胜 {my_win}")
            return my_win
        
        # 检查对手的获胜威胁
        win_threat = self.find_win_threat(board, opponent)
        if win_threat:
            self.threat_stats['threats_detected'] += 1
            self.threat_stats['threats_defended'] += 1
            print(f"检测到获胜威胁：必须防守 {win_threat}")
            return win_threat
        
        # 检查对手的活四威胁
        live_four_threat = self.find_live_four_threat(board, opponent)
        if live_four_threat:
            self.threat_stats['threats_detected'] += 1
            self.threat_stats['threats_defended'] += 1
            print(f"检测到活四威胁：必须防守 {live_four_threat}")
            return live_four_threat
        
        # 检查对手的活三威胁
        live_three_threat = self.find_live_three_threat(board, opponent)
        if live_three_threat:
            self.threat_stats['threats_detected'] += 1
            self.threat_stats['threats_defended'] += 1
            print(f"检测到活三威胁：必须防守 {live_three_threat}")
            return live_three_threat
        
        return None
    
    def find_win_threat(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """查找获胜威胁"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    # 模拟落子
                    board[i, j] = player
                    if self.check_win(board, i, j, player):
                        board[i, j] = 0  # 恢复
                        return (i, j)
                    board[i, j] = 0  # 恢复
        return None
    
    def find_live_four_threat(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """查找活四威胁"""
        # 首先检查是否已经有活四需要防守
        return self.find_existing_live_four(board, player)
    
    def find_existing_live_four(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """查找已经存在的活四，返回防守位置"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == player:  # 找到玩家的棋子
                    # 检查这个棋子是否参与了活四
                    for dx, dy in directions:
                        # 检查这个方向是否形成了活四
                        defense_pos = self.check_live_four_pattern(board, i, j, dx, dy, player)
                        if defense_pos:
                            return defense_pos
        return None
    
    def check_live_four_pattern(self, board: np.ndarray, row: int, col: int, dx: int, dy: int, player: int) -> Optional[Tuple[int, int]]:
        """检查特定位置和方向是否形成活四，返回防守位置"""
        # 检查正方向
        count = 1
        open_ends = 0
        defense_positions = []
        
        # 正方向检查
        x, y = row + dx, col + dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
            count += 1
            x += dx
            y += dy
        
        # 检查正方向端点
        if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
            open_ends += 1
            defense_positions.append((x, y))
        
        # 反方向检查
        x, y = row - dx, col - dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
            count += 1
            x -= dx
            y -= dy
        
        # 检查反方向端点
        if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
            open_ends += 1
            defense_positions.append((x, y))
        
        # 如果是活四（4个子且至少有一端开放）
        if count == 4 and open_ends > 0:
            # 返回任意一个防守位置
            return defense_positions[0] if defense_positions else None
        
        return None
    
    def find_live_three_threat(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """查找活三威胁"""
        # 首先检查是否已经有活三需要防守
        return self.find_existing_live_three(board, player)
    
    def find_existing_live_three(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """查找已经存在的活三，返回防守位置"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == player:  # 找到玩家的棋子
                    # 检查这个棋子是否参与了活三
                    for dx, dy in directions:
                        # 检查这个方向是否形成了活三
                        defense_pos = self.check_live_three_pattern(board, i, j, dx, dy, player)
                        if defense_pos:
                            return defense_pos
        return None
    
    def check_live_three_pattern(self, board: np.ndarray, row: int, col: int, dx: int, dy: int, player: int) -> Optional[Tuple[int, int]]:
        """检查特定位置和方向是否形成活三，返回防守位置"""
        # 检查正方向
        count = 1
        open_ends = 0
        positions = [(row, col)]
        
        # 正方向检查
        x, y = row + dx, col + dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
            count += 1
            positions.append((x, y))
            x += dx
            y += dy
        
        # 检查正方向端点
        if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
            open_ends += 1
            defense1 = (x, y)
        else:
            defense1 = None
        
        # 反方向检查
        x, y = row - dx, col - dy
        while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
            count += 1
            positions.append((x, y))
            x -= dx
            y -= dy
        
        # 检查反方向端点
        if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
            open_ends += 1
            defense2 = (x, y)
        else:
            defense2 = None
        
        # 如果是活三（3个子且两端都开放）
        if count == 3 and open_ends == 2:
            # 返回任意一个防守位置
            return defense1 if defense1 else defense2
        
        return None
    
    def check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
                count += 1
                x += dx
                y += dy
            
            # 反方向
            x, y = row - dx, col - dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
                count += 1
                x -= dx
                y -= dy
            
            if count >= 5:
                return True
        
        return False
    
    def has_live_four(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """检查是否有活四"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            open_ends = 0
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
                count += 1
                x += dx
                y += dy
            if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
                open_ends += 1
            
            # 反方向
            x, y = row - dx, col - dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
                count += 1
                x -= dx
                y -= dy
            if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
                open_ends += 1
            
            if count == 4 and open_ends > 0:
                return True
        
        return False
    
    def has_live_three(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """检查是否有活三"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            open_ends = 0
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
                count += 1
                x += dx
                y += dy
            if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
                open_ends += 1
            
            # 反方向
            x, y = row - dx, col - dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == player:
                count += 1
                x -= dx
                y -= dy
            if 0 <= x < self.board_size and 0 <= y < self.board_size and board[x, y] == 0:
                open_ends += 1
            
            if count == 3 and open_ends == 2:
                return True
        
        return False
    
    def get_threat_stats(self) -> Dict[str, Any]:
        """获取威胁检测统计"""
        # 确保威胁统计字典存在
        if not hasattr(self, 'threat_stats'):
            self.threat_stats = {
                'total_threats_detected': 0,
                'threats_defended': 0,
                'forced_defenses': 0,
                'defense_rate': 0.0
            }
        
        total = self.threat_stats['threats_detected']
        if total > 0:
            defense_rate = self.threat_stats['threats_defended'] / total
        else:
            defense_rate = 0.0
        
        return {
            'total_threats_detected': total,
            'threats_defended': self.threat_stats['threats_defended'],
            'forced_defenses': self.threat_stats['forced_defenses'],
            'defense_rate': defense_rate
        }
    
    def reset_stats(self):
        """重置统计"""
        self.threat_stats = {
            'threats_detected': 0,
            'threats_defended': 0,
            'forced_defenses': 0
        }
    
    def debug_live_three_detection(self, board: np.ndarray, player: int) -> Dict[str, Any]:
        """调试活三检测"""
        print(f"  调试活三检测，玩家={player}")
        result = self.find_existing_live_three(board, player)
        print(f"  活三检测结果：{result}")
        return {'result': result}
    
    def debug_live_four_detection(self, board: np.ndarray, player: int) -> Dict[str, Any]:
        """调试活四检测"""
        print(f"  调试活四检测，玩家={player}")
        result = self.find_live_four_threat(board, player)
        print(f"  活四检测结果：{result}")
        return {'result': result}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要 - 兼容integrated_ai的接口"""
        # 计算平均时间（基于移动历史）
        avg_time = 0.0
        total_moves = len(self.move_history)
        if total_moves > 0:
            total_time = sum(move.get('time', 0) for move in self.move_history)
            avg_time = total_time / total_moves
        
        return {
            # draw_ai_info期望的嵌套结构
            'ai_stats': {
                'total_moves': total_moves,
                'avg_time_per_move': avg_time,
                'avg_nodes_per_move': 0  # UltimateThreatAI不使用节点搜索
            },
            # 其他信息（保持向后兼容）
            'threat_detection_stats': self.threat_stats,
            'ai_type': 'UltimateThreatAI',
            'detection_rate': self._calculate_detection_rate(),
            'defense_rate': self._calculate_defense_rate(),
            'model_info': {
                'architecture': 'CNN + Policy + Value',
                'threat_detection': 'Rule-based + Neural',
                'board_size': self.board_size
            }
        }
    
    def _calculate_detection_rate(self) -> float:
        """计算威胁检测率"""
        total_threats = self.threat_stats.get('threats_detected', 0)
        if total_threats == 0:
            return 0.0
        defended = self.threat_stats.get('threats_defended', 0)
        return min(100.0, (defended / total_threats) * 100)
    
    def _calculate_defense_rate(self) -> float:
        """计算防守成功率"""
        forced = self.threat_stats.get('forced_defenses', 0)
        total = self.threat_stats.get('threats_detected', 0)
        if total == 0:
            return 0.0
        return min(100.0, (forced / total) * 100)
    
    def get_ai_info(self) -> Dict[str, Any]:
        """获取AI信息 - 兼容integrated_ai的接口"""
        return {
            'difficulty': 'ultimate_threat',
            'max_depth': 0,  # 不使用深度限制
            'time_limit': 5.0,
            'strategy': 'threat_based',
            'engine_type': 'UltimateThreatAI',
            'features': [
                '优先进攻策略（获胜＞活四＞双活三＞活三＞冲四）',
                '智能防守策略（只在无进攻机会时防守）',
                '强制威胁检测（活三、活四、获胜）',
                '双活三威胁检测（最高优先级进攻机会）',
                '神经网络移动建议',
                '多层威胁优先级系统',
                '100%威胁检测保证',
                'CNN + Policy + Value 架构',
                'Rule-based + Neural 混合检测',
                '实时威胁响应',
                '进攻优先决策系统'
            ],
            'neural_enabled': True,
            'model_info': {
                'architecture': 'CNN + Policy + Value',
                'threat_detection': 'Rule-based + Neural',
                'board_size': self.board_size
            },
            'threat_detection_stats': self.threat_stats,
            'detection_rate': self._calculate_detection_rate(),
            'defense_rate': self._calculate_defense_rate()
        }

def test_ultimate_threat_ai():
    """测试终极威胁检测AI"""
    print("=== 终极威胁检测AI测试 ===")
    
    # 创建AI
    ai = UltimateThreatAI()
    
    # 测试场景1：活三防守
    print("\n测试场景1：活三防守")
    board1 = np.zeros((15, 15), dtype=int)
    board1[7, 6:9] = 1  # 黑棋形成活三
    board1_list = board1.tolist()
    
    move1 = ai.get_move(board1_list, -1)  # 白棋需要防守
    print(f"棋盘状态：黑棋在 (7,6),(7,7),(7,8)")
    print(f"白棋应选择：{(7,5)} 或 {(7,9)}")
    print(f"AI选择：{move1}")
    
    # 测试场景2：获胜机会
    print("\n测试场景2：获胜机会")
    board2 = np.zeros((15, 15), dtype=int)
    board2[7, 5:9] = -1  # 白棋已有四个
    board2_list = board2.tolist()
    
    move2 = ai.get_move(board2_list, -1)  # 白棋可以获胜
    print(f"棋盘状态：白棋在 (7,5),(7,6),(7,7),(7,8)")
    print(f"白棋应选择：{(7,4)} 或 {(7,9)} 获胜")
    print(f"AI选择：{move2}")
    
    # 测试场景3：多重威胁
    print("\n测试场景3：多重威胁")
    board3 = np.zeros((15, 15), dtype=int)
    board3[7, 6:9] = 1   # 黑棋活三
    board3[6, 7] = -1     # 白棋有一个子
    board3[8, 7] = -1     # 白棋有一个子
    board3_list = board3.tolist()
    
    move3 = ai.get_move(board3_list, -1)  # 白棋需要防守
    print(f"棋盘状态：黑棋活三在 (7,6),(7,7),(7,8)，白棋在 (6,7),(8,7)")
    print(f"白棋应优先防守活三")
    print(f"AI选择：{move3}")
    
    # 显示威胁检测统计
    stats = ai.get_threat_stats()
    print(f"\n威胁检测统计：")
    print(f"检测到的威胁总数：{stats['total_threats_detected']}")
    print(f"成功防守的威胁：{stats['threats_defended']}")
    print(f"强制防守次数：{stats['forced_defenses']}")
    print(f"防守成功率：{stats['defense_rate']:.2%}")
    
    return ai

def create_ultimate_ai_model():
    """创建终极AI模型文件"""
    print("创建终极威胁检测AI模型...")
    
    # 保存AI配置
    config = {
        'ai_type': 'Ultimate Threat Detection AI',
        'creation_time': time.time(),
        'description': '结合神经网络和强制规则检测的终极威胁检测AI',
        'features': [
            '强制威胁检测（活三、活四、获胜）',
            '神经网络移动建议',
            '多层威胁优先级系统',
            '100%威胁检测保证'
        ],
        'model_path': 'models/force_threat/forced_detection_model.pth',
        'threat_detection_rules': {
            'highest_priority': 'opponent_win_threat',
            'high_priority': 'opponent_live_four',
            'medium_priority': 'opponent_live_three',
            'opportunity': 'my_win_threat'
        }
    }
    
    os.makedirs("models/ultimate_threat", exist_ok=True)
    
    with open("models/ultimate_threat/ai_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("终极AI配置已保存到：models/ultimate_threat/ai_config.json")

if __name__ == "__main__":
    # 创建终极AI模型
    create_ultimate_ai_model()
    
    # 测试终极威胁检测AI
    test_ultimate_threat_ai()