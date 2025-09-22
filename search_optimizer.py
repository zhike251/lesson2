"""
搜索优化模块
实现启发式搜索排序、历史启发表、killer moves和时间控制

作者：Claude AI Engineer
日期：2025-09-22
"""

import time
import random
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
import heapq

# 游戏常量
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

class SearchStrategy(Enum):
    """搜索策略枚举"""
    NORMAL = 0
    AGGRESSIVE = 1
    DEFENSIVE = 2
    BALANCED = 3

@dataclass
class MoveInfo:
    """移动信息"""
    position: Tuple[int, int]
    score: int
    heuristic_score: int
    pattern_types: List[str]
    threat_level: int
    
    def __lt__(self, other):
        return self.score > other.score  # 降序排序

@dataclass
class SearchStats:
    """搜索统计信息"""
    total_nodes: int
    alpha_beta_cutoffs: int
    killer_move_hits: int
    history_table_hits: int
    transposition_table_hits: int
    search_depth: int
    time_elapsed: float
    nodes_per_second: float
    
class HeuristicSearchOrder:
    """启发式搜索排序"""
    
    def __init__(self):
        # 棋型优先级
        self.pattern_priority = {
            'FIVE': 100000,
            'OPEN_FOUR': 10000,
            'BLOCKED_FOUR': 5000,
            'OPEN_THREE': 1000,
            'BLOCKED_THREE': 100,
            'OPEN_TWO': 200,
            'BLOCKED_TWO': 50,
            'SINGLE': 10,
        }
        
        # 威胁优先级
        self.threat_priority = {
            'WINNING': 100000,
            'CRITICAL': 10000,
            'HIGH': 1000,
            'MEDIUM': 100,
            'LOW': 10,
            'NONE': 0,
        }
        
    def rank_moves(self, board: List[List[int]], moves: List[Tuple[int, int]], 
                   player: int, strategy: SearchStrategy = SearchStrategy.BALANCED) -> List[MoveInfo]:
        """对移动进行排序"""
        move_infos = []
        
        for move in moves:
            row, col = move
            if board[row][col] != EMPTY:
                continue
            
            # 计算启发式分数
            heuristic_score = self._calculate_heuristic_score(board, row, col, player, strategy)
            
            # 识别棋型
            pattern_types = self._identify_patterns(board, row, col, player)
            
            # 评估威胁等级
            threat_level = self._evaluate_threat_level(board, row, col, player)
            
            # 综合分数
            total_score = heuristic_score
            
            # 根据策略调整分数
            if strategy == SearchStrategy.AGGRESSIVE:
                total_score += self._calculate_aggressive_bonus(pattern_types, threat_level)
            elif strategy == SearchStrategy.DEFENSIVE:
                total_score += self._calculate_defensive_bonus(board, row, col, player)
            
            move_info = MoveInfo(
                position=move,
                score=total_score,
                heuristic_score=heuristic_score,
                pattern_types=pattern_types,
                threat_level=threat_level
            )
            
            move_infos.append(move_info)
        
        # 按分数排序
        move_infos.sort()
        
        return move_infos
    
    def _calculate_heuristic_score(self, board: List[List[int]], row: int, col: int, 
                                  player: int, strategy: SearchStrategy) -> int:
        """计算启发式分数"""
        score = 0
        
        # 位置权重
        center_distance = abs(row - 7) + abs(col - 7)
        score += (14 - center_distance) * 5
        
        # 连通性分数
        connectivity_score = self._calculate_connectivity_score(board, row, col, player)
        score += connectivity_score
        
        # 控制力分数
        control_score = self._calculate_control_score(board, row, col, player)
        score += control_score
        
        # 边缘惩罚
        edge_penalty = self._calculate_edge_penalty(row, col)
        score -= edge_penalty
        
        return score
    
    def _calculate_connectivity_score(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """计算连通性分数"""
        connectivity_score = 0
        
        # 检查四个方向的连通性
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            connected_count = 0
            empty_count = 0
            
            # 正方向
            x, y = row + dx, col + dy
            for _ in range(4):
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    if board[x][y] == player:
                        connected_count += 1
                    elif board[x][y] == EMPTY:
                        empty_count += 1
                    else:
                        break
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            for _ in range(4):
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    if board[x][y] == player:
                        connected_count += 1
                    elif board[x][y] == EMPTY:
                        empty_count += 1
                    else:
                        break
                x -= dx
                y -= dy
            
            # 连通性分数
            if connected_count >= 2:
                connectivity_score += connected_count * 20
            if empty_count >= 2:
                connectivity_score += empty_count * 5
        
        return connectivity_score
    
    def _calculate_control_score(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """计算控制力分数"""
        control_score = 0
        
        # 检查周围3x3区域
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = row + di, col + dj
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                    if board[ni][nj] == player:
                        control_score += 30
                    elif board[ni][nj] == 3 - player:
                        control_score -= 15
        
        return control_score
    
    def _calculate_edge_penalty(self, row: int, col: int) -> int:
        """计算边缘惩罚"""
        penalty = 0
        
        # 边缘惩罚
        if row <= 1 or row >= BOARD_SIZE - 2:
            penalty += 10
        if col <= 1 or col >= BOARD_SIZE - 2:
            penalty += 10
        
        # 角落惩罚
        if (row <= 1 and col <= 1) or (row <= 1 and col >= BOARD_SIZE - 2) or \
           (row >= BOARD_SIZE - 2 and col <= 1) or (row >= BOARD_SIZE - 2 and col >= BOARD_SIZE - 2):
            penalty += 20
        
        return penalty
    
    def _identify_patterns(self, board: List[List[int]], row: int, col: int, player: int) -> List[str]:
        """识别棋型"""
        patterns = []
        
        # 模拟落子
        board[row][col] = player
        
        # 检查各个方向的棋型
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            pattern = self._check_pattern_in_direction(board, row, col, dx, dy, player)
            if pattern:
                patterns.append(pattern)
        
        # 恢复棋盘
        board[row][col] = EMPTY
        
        return patterns
    
    def _check_pattern_in_direction(self, board: List[List[int]], row: int, col: int, 
                                   dx: int, dy: int, player: int) -> Optional[str]:
        """检查某个方向的棋型"""
        line = []
        
        # 获取该方向的线
        x, y = row, col
        for _ in range(5):
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                line.append(board[x][y])
            else:
                line.append(-1)
            x += dx
            y += dy
        
        line_str = ''.join(map(str, line))
        
        # 检查棋型
        if line_str.startswith('22222') or line_str.startswith('11111'):
            return 'FIVE'
        elif '022220' in line_str or '011110' in line_str:
            return 'OPEN_FOUR'
        elif '02222' in line_str or '22220' in line_str or '01111' in line_str or '11110' in line_str:
            return 'BLOCKED_FOUR'
        elif '02220' in line_str or '01110' in line_str:
            return 'OPEN_THREE'
        elif '0222' in line_str or '2220' in line_str or '0111' in line_str or '1110' in line_str:
            return 'BLOCKED_THREE'
        elif '0220' in line_str or '0110' in line_str:
            return 'OPEN_TWO'
        elif '022' in line_str or '220' in line_str or '011' in line_str or '110' in line_str:
            return 'BLOCKED_TWO'
        
        return None
    
    def _evaluate_threat_level(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估威胁等级"""
        threat_level = 0
        
        # 模拟落子
        board[row][col] = player
        
        # 检查是否能获胜
        if self._check_win(board, row, col, player):
            threat_level = 5
        # 检查是否能形成活四
        elif self._has_open_four(board, row, col, player):
            threat_level = 4
        # 检查是否能形成活三
        elif self._has_open_three(board, row, col, player):
            threat_level = 3
        # 检查是否能形成活二
        elif self._has_open_two(board, row, col, player):
            threat_level = 2
        else:
            threat_level = 1
        
        # 恢复棋盘
        board[row][col] = EMPTY
        
        return threat_level
    
    def _check_win(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
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
    
    def _has_open_four(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否有活四"""
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            line = []
            # 获取该方向的线
            for direction in [-1, 1]:
                x, y = row, col
                for _ in range(5):
                    if direction == -1:
                        x -= dx
                        y -= dy
                    else:
                        if direction == 1 and (x != row or y != col):
                            x += dx
                            y += dy
                    
                    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                        if direction == -1:
                            line.insert(0, board[x][y])
                        else:
                            line.append(board[x][y])
                    else:
                        if direction == -1:
                            line.insert(0, -1)
                        else:
                            line.append(-1)
            
            line_str = ''.join(map(str, line))
            if '022220' in line_str or '011110' in line_str:
                return True
        
        return False
    
    def _has_open_three(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否有活三"""
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            line = []
            # 获取该方向的线
            for direction in [-1, 1]:
                x, y = row, col
                for _ in range(4):
                    if direction == -1:
                        x -= dx
                        y -= dy
                    else:
                        if direction == 1 and (x != row or y != col):
                            x += dx
                            y += dy
                    
                    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                        if direction == -1:
                            line.insert(0, board[x][y])
                        else:
                            line.append(board[x][y])
                    else:
                        if direction == -1:
                            line.insert(0, -1)
                        else:
                            line.append(-1)
            
            line_str = ''.join(map(str, line))
            if '02220' in line_str or '01110' in line_str:
                return True
        
        return False
    
    def _has_open_two(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否有活二"""
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            line = []
            # 获取该方向的线
            for direction in [-1, 1]:
                x, y = row, col
                for _ in range(3):
                    if direction == -1:
                        x -= dx
                        y -= dy
                    else:
                        if direction == 1 and (x != row or y != col):
                            x += dx
                            y += dy
                    
                    if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                        if direction == -1:
                            line.insert(0, board[x][y])
                        else:
                            line.append(board[x][y])
                    else:
                        if direction == -1:
                            line.insert(0, -1)
                        else:
                            line.append(-1)
            
            line_str = ''.join(map(str, line))
            if '0220' in line_str or '0110' in line_str:
                return True
        
        return False
    
    def _calculate_aggressive_bonus(self, pattern_types: List[str], threat_level: int) -> int:
        """计算进攻奖励"""
        bonus = 0
        
        # 高威胁等级奖励
        bonus += threat_level * 100
        
        # 进攻性棋型奖励
        for pattern in pattern_types:
            if pattern in ['FIVE', 'OPEN_FOUR', 'OPEN_THREE']:
                bonus += self.pattern_priority.get(pattern, 0)
        
        return bonus
    
    def _calculate_defensive_bonus(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """计算防守奖励"""
        bonus = 0
        opponent = 3 - player
        
        # 检查是否阻止对手的威胁
        board[row][col] = opponent
        
        # 如果对手在这里能获胜，给予高分
        if self._check_win(board, row, col, opponent):
            bonus += 50000
        # 如果对手在这里能形成活四，给予高分
        elif self._has_open_four(board, row, col, opponent):
            bonus += 10000
        # 如果对手在这里能形成活三，给予中等分
        elif self._has_open_three(board, row, col, opponent):
            bonus += 1000
        
        board[row][col] = EMPTY
        
        return bonus

class HistoryHeuristicTable:
    """历史启发表"""
    
    def __init__(self):
        self.history_table = defaultdict(int)
        self.max_entries = 10000
        
    def update(self, move: Tuple[int, int], depth: int, score: int):
        """更新历史启发表"""
        # 使用深度和分数的乘积作为更新值
        update_value = depth * depth * abs(score)
        self.history_table[move] += update_value
        
        # 限制表的大小
        if len(self.history_table) > self.max_entries:
            # 删除最旧的条目
            oldest_key = min(self.history_table.keys(), key=lambda k: self.history_table[k])
            del self.history_table[oldest_key]
    
    def get_score(self, move: Tuple[int, int]) -> int:
        """获取移动的历史分数"""
        return self.history_table.get(move, 0)
    
    def get_top_moves(self, n: int = 10) -> List[Tuple[Tuple[int, int], int]]:
        """获取历史分数最高的移动"""
        sorted_moves = sorted(self.history_table.items(), key=lambda x: x[1], reverse=True)
        return sorted_moves[:n]
    
    def clear(self):
        """清空历史表"""
        self.history_table.clear()

class KillerMoveTable:
    """Killer Move表"""
    
    def __init__(self):
        self.killer_moves = [[None for _ in range(2)] for _ in range(20)]  # 20层深度，每层2个killer moves
        self.killer_counts = defaultdict(int)
        
    def add_killer_move(self, move: Tuple[int, int], depth: int):
        """添加killer move"""
        if depth < len(self.killer_moves):
            # 如果移动已经在表中，增加计数
            if move in self.killer_moves[depth]:
                self.killer_counts[move] += 1
            else:
                # 否则添加到表中
                self.killer_moves[depth][1] = self.killer_moves[depth][0]
                self.killer_moves[depth][0] = move
                self.killer_counts[move] = 1
    
    def is_killer_move(self, move: Tuple[int, int], depth: int) -> bool:
        """检查是否是killer move"""
        if depth < len(self.killer_moves):
            return move in self.killer_moves[depth]
        return False
    
    def get_killer_moves(self, depth: int) -> List[Tuple[int, int]]:
        """获取指定深度的killer moves"""
        if depth < len(self.killer_moves):
            return [move for move in self.killer_moves[depth] if move is not None]
        return []
    
    def get_top_killer_moves(self, n: int = 10) -> List[Tuple[Tuple[int, int], int]]:
        """获取最有效的killer moves"""
        sorted_killers = sorted(self.killer_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_killers[:n]

class TranspositionTable:
    """置换表"""
    
    def __init__(self, max_size: int = 1000000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def store(self, board_key: int, depth: int, score: int, move: Tuple[int, int], 
              flag: str, age: int):
        """存储搜索结果"""
        if len(self.table) >= self.max_size:
            # 删除最旧的条目
            oldest_key = min(self.table.keys(), key=lambda k: self.table[k].get('age', 0))
            del self.table[oldest_key]
        
        self.table[board_key] = {
            'depth': depth,
            'score': score,
            'move': move,
            'flag': flag,  # 'EXACT', 'LOWER', 'UPPER'
            'age': age
        }
    
    def lookup(self, board_key: int, depth: int, alpha: float, beta: float) -> Optional[Dict]:
        """查找搜索结果"""
        if board_key in self.table:
            entry = self.table[board_key]
            if entry['depth'] >= depth:
                self.hits += 1
                
                if entry['flag'] == 'EXACT':
                    return entry
                elif entry['flag'] == 'LOWER' and entry['score'] >= beta:
                    return entry
                elif entry['flag'] == 'UPPER' and entry['score'] <= alpha:
                    return entry
            else:
                self.misses += 1
        else:
            self.misses += 1
        
        return None
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    def clear(self):
        """清空置换表"""
        self.table.clear()
        self.hits = 0
        self.misses = 0

class TimeController:
    """时间控制器"""
    
    def __init__(self, time_limit: float = 5.0):
        self.time_limit = time_limit
        self.start_time = 0
        self.time_check_interval = 1000  # 每搜索1000个节点检查一次时间
        self.node_count = 0
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.node_count = 0
    
    def check_time(self) -> bool:
        """检查时间是否用完"""
        self.node_count += 1
        
        # 定期检查时间
        if self.node_count % self.time_check_interval == 0:
            elapsed = time.time() - self.start_time
            return elapsed < self.time_limit
        
        return True
    
    def time_elapsed(self) -> float:
        """获取已用时间"""
        return time.time() - self.start_time
    
    def time_remaining(self) -> float:
        """获取剩余时间"""
        return max(0, self.time_limit - self.time_elapsed())

class SearchOptimizer:
    """搜索优化器"""
    
    def __init__(self, time_limit: float = 5.0):
        self.heuristic_search = HeuristicSearchOrder()
        self.history_table = HistoryHeuristicTable()
        self.killer_table = KillerMoveTable()
        self.transposition_table = TranspositionTable()
        self.time_controller = TimeController(time_limit)
        
        # 搜索统计
        self.stats = SearchStats(
            total_nodes=0,
            alpha_beta_cutoffs=0,
            killer_move_hits=0,
            history_table_hits=0,
            transposition_table_hits=0,
            search_depth=0,
            time_elapsed=0,
            nodes_per_second=0
        )
    
    def optimize_move_ordering(self, board: List[List[int]], moves: List[Tuple[int, int]], 
                             player: int, depth: int, strategy: SearchStrategy = SearchStrategy.BALANCED) -> List[Tuple[int, int]]:
        """优化移动顺序"""
        # 1. 启发式排序
        move_infos = self.heuristic_search.rank_moves(board, moves, player, strategy)
        
        # 2. 应用历史启发
        for move_info in move_infos:
            history_score = self.history_table.get_score(move_info.position)
            move_info.score += history_score
            
            if history_score > 0:
                self.stats.history_table_hits += 1
        
        # 3. 应用killer moves
        killer_moves = self.killer_table.get_killer_moves(depth)
        for killer_move in killer_moves:
            for move_info in move_infos:
                if move_info.position == killer_move:
                    move_info.score += 1000
                    self.stats.killer_move_hits += 1
                    break
        
        # 4. 重新排序
        move_infos.sort()
        
        return [move_info.position for move_info in move_infos]
    
    def update_tables(self, move: Tuple[int, int], depth: int, score: int):
        """更新各种表"""
        # 更新历史启发表
        self.history_table.update(move, depth, score)
        
        # 更新killer move表
        if abs(score) > 1000:  # 认为是好移动
            self.killer_table.add_killer_move(move, depth)
    
    def start_search(self):
        """开始搜索"""
        self.time_controller.start()
        self.stats.total_nodes = 0
        self.stats.alpha_beta_cutoffs = 0
        self.stats.killer_move_hits = 0
        self.stats.history_table_hits = 0
        self.stats.transposition_table_hits = 0
    
    def should_continue_search(self) -> bool:
        """判断是否应该继续搜索"""
        return self.time_controller.check_time()
    
    def finalize_search(self):
        """结束搜索"""
        self.stats.time_elapsed = self.time_controller.time_elapsed()
        self.stats.nodes_per_second = self.stats.total_nodes / max(1, self.stats.time_elapsed)
    
    def get_search_stats(self) -> SearchStats:
        """获取搜索统计"""
        return self.stats
    
    def clear_tables(self):
        """清空所有表"""
        self.history_table.clear()
        self.killer_table.killer_counts.clear()
        self.transposition_table.clear()

# 测试函数
def test_search_optimizer():
    """测试搜索优化器"""
    print("正在测试搜索优化器...")
    
    # 创建搜索优化器
    optimizer = SearchOptimizer(time_limit=2.0)
    
    # 创建测试棋盘
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # 设置一些测试棋子
    board[7][7] = BLACK
    board[7][8] = WHITE
    board[8][7] = WHITE
    
    # 获取候选移动
    candidates = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == EMPTY:
                candidates.append((i, j))
    
    # 测试移动排序
    optimized_moves = optimizer.optimize_move_ordering(board, candidates, BLACK, 3)
    
    print(f"优化后的移动顺序前5位: {optimized_moves[:5]}")
    
    # 更新表
    if optimized_moves:
        optimizer.update_tables(optimized_moves[0], 3, 1000)
    
    # 获取统计信息
    stats = optimizer.get_search_stats()
    print(f"搜索统计: {stats}")
    
    print("搜索优化器测试完成！")

if __name__ == "__main__":
    test_search_optimizer()