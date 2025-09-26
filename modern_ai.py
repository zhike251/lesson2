"""
现代化五子棋AI引擎
基于OpenSpiel最佳实践实现的高性能五子棋AI系统

作者：Claude AI Engineer
日期：2025-09-22
"""

import time
import random
import math
from typing import List, Tuple, Optional, Dict, Set, Any
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import sys

# 游戏常量
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

# 棋型权重（基于专业五子棋AI的标准权重）
PATTERN_WEIGHTS = {
    'FIVE': 100000,           # 五连
    'OPEN_FOUR': 10000,       # 活四
    'DOUBLE_THREE': 5000,     # 双活三
    'OPEN_THREE': 1000,       # 活三
    'BLOCKED_FOUR': 500,      # 冲四
    'OPEN_TWO': 200,          # 活二
    'BLOCKED_THREE': 100,     # 眠三
    'BLOCKED_TWO': 50,        # 眠二
    'SINGLE': 10,             # 单子
}

# 方向向量
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

class PatternType(Enum):
    """棋型枚举"""
    NONE = 0
    SINGLE = 1
    BLOCKED_TWO = 2
    OPEN_TWO = 3
    BLOCKED_THREE = 4
    OPEN_THREE = 5
    BLOCKED_FOUR = 6
    OPEN_FOUR = 7
    FIVE = 8
    DOUBLE_THREE = 9

@dataclass
class Pattern:
    """棋型数据类"""
    pattern_type: PatternType
    count: int
    score: int

@dataclass
class SearchResult:
    """搜索结果数据类"""
    score: int
    move: Optional[Tuple[int, int]]
    nodes_searched: int
    time_elapsed: float
    depth: int
    alpha_beta_cutoffs: int

class AdvancedEvaluator:
    """高级评估函数"""
    
    def __init__(self):
        # 位置权重矩阵（中心位置权重更高）
        self.position_weights = self._create_position_weights()
        
        # 棋型模式字典
        self.pattern_dict = self._create_pattern_dict()
        
    def _create_position_weights(self) -> List[List[int]]:
        """创建位置权重矩阵"""
        weights = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        center = BOARD_SIZE // 2
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                # 计算到中心的距离
                distance = abs(i - center) + abs(j - center)
                # 中心位置权重最高，边缘权重最低
                weights[i][j] = max(1, 15 - distance)
                
        return weights
    
    def _create_pattern_dict(self) -> Dict[str, PatternType]:
        """创建棋型模式字典"""
        patterns = {}
        
        # 五连模式
        patterns['22222'] = PatternType.FIVE
        patterns['11111'] = PatternType.FIVE
        
        # 活四模式
        patterns['022220'] = PatternType.OPEN_FOUR
        patterns['011110'] = PatternType.OPEN_FOUR
        
        # 冲四模式
        patterns['02222'] = PatternType.BLOCKED_FOUR
        patterns['22220'] = PatternType.BLOCKED_FOUR
        patterns['01111'] = PatternType.BLOCKED_FOUR
        patterns['11110'] = PatternType.BLOCKED_FOUR
        patterns['20222'] = PatternType.BLOCKED_FOUR
        patterns['22022'] = PatternType.BLOCKED_FOUR
        patterns['22202'] = PatternType.BLOCKED_FOUR
        patterns['10111'] = PatternType.BLOCKED_FOUR
        patterns['11011'] = PatternType.BLOCKED_FOUR
        patterns['11101'] = PatternType.BLOCKED_FOUR
        
        # 活三模式
        patterns['02220'] = PatternType.OPEN_THREE
        patterns['01110'] = PatternType.OPEN_THREE
        patterns['002220'] = PatternType.OPEN_THREE
        patterns['011100'] = PatternType.OPEN_THREE
        
        # 眠三模式
        patterns['0222'] = PatternType.BLOCKED_THREE
        patterns['2220'] = PatternType.BLOCKED_THREE
        patterns['0111'] = PatternType.BLOCKED_THREE
        patterns['1110'] = PatternType.BLOCKED_THREE
        patterns['2022'] = PatternType.BLOCKED_THREE
        patterns['2202'] = PatternType.BLOCKED_THREE
        patterns['02022'] = PatternType.BLOCKED_THREE
        patterns['02202'] = PatternType.BLOCKED_THREE
        
        # 活二模式
        patterns['0220'] = PatternType.OPEN_TWO
        patterns['0110'] = PatternType.OPEN_TWO
        patterns['00220'] = PatternType.OPEN_TWO
        patterns['00110'] = PatternType.OPEN_TWO
        
        # 眠二模式
        patterns['022'] = PatternType.BLOCKED_TWO
        patterns['220'] = PatternType.BLOCKED_TWO
        patterns['011'] = PatternType.BLOCKED_TWO
        patterns['110'] = PatternType.BLOCKED_TWO
        
        return patterns
    
    def evaluate_board(self, board: List[List[int]], player: int) -> int:
        """评估整个棋盘状态"""
        score = 0
        opponent = 3 - player
        
        # 评估所有方向的棋型
        for dx, dy in DIRECTIONS:
            score += self._evaluate_direction(board, player, dx, dy)
            score -= self._evaluate_direction(board, opponent, dx, dy) * 1.1  # 防守权重略高
        
        return score
    
    def _evaluate_direction(self, board: List[List[int]], player: int, dx: int, dy: int) -> int:
        """评估某个方向的棋型"""
        score = 0
        visited = set()
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if (i, j) not in visited and board[i][j] == player:
                    # 获取该位置的棋型
                    line = self._get_line(board, i, j, dx, dy, player)
                    if line:
                        pattern = self._analyze_pattern(line, player)
                        score += pattern.score
                        
                        # 标记已访问的位置
                        for k in range(len(line)):
                            if line[k] == player:
                                x, y = i + k * dx, j + k * dy
                                visited.add((x, y))
        
        return score
    
    def _get_line(self, board: List[List[int]], row: int, col: int, dx: int, dy: int, player: int) -> List[int]:
        """获取某个方向的连续棋子线"""
        line = []
        
        # 向正方向延伸
        x, y = row, col
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            line.append(board[x][y])
            x += dx
            y += dy
        
        # 如果线太短，返回空
        if len(line) < 3:
            return []
        
        return line
    
    def _analyze_pattern(self, line: List[int], player: int) -> Pattern:
        """分析棋型"""
        line_str = ''.join(map(str, line))
        
        # 查找匹配的棋型
        for pattern_str, pattern_type in self.pattern_dict.items():
            if pattern_str in line_str:
                return Pattern(
                    pattern_type=pattern_type,
                    count=1,
                    score=PATTERN_WEIGHTS.get(pattern_type.name, 0)
                )
        
        # 如果没有匹配的棋型，返回单子
        player_count = line_str.count(str(player))
        return Pattern(
            pattern_type=PatternType.SINGLE,
            count=player_count,
            score=PATTERN_WEIGHTS['SINGLE'] * player_count
        )
    
    def evaluate_move(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估某个位置的分数"""
        if board[row][col] != EMPTY:
            return -float('inf')
        
        score = 0
        opponent = 3 - player
        
        # 模拟落子
        board[row][col] = player
        player_score = self.evaluate_board(board, player)
        board[row][col] = EMPTY
        
        # 模拟对手落子
        board[row][col] = opponent
        opponent_score = self.evaluate_board(board, opponent)
        board[row][col] = EMPTY
        
        # 综合评分（进攻 + 防守）
        score = player_score + opponent_score * 1.2
        
        # 添加位置权重
        score += self.position_weights[row][col] * 10
        
        return score

class SearchOptimizer:
    """搜索优化器"""
    
    def __init__(self):
        # 历史启发表
        self.history_table = defaultdict(int)
        
        # Killer moves
        self.killer_moves = [[None for _ in range(2)] for _ in range(20)]
        
        # 威胁检测缓存
        self.threat_cache = {}
        
    def get_move_order(self, board: List[List[int]], moves: List[Tuple[int, int]], 
                      player: int, depth: int) -> List[Tuple[int, int]]:
        """获取移动顺序（启发式排序）"""
        scored_moves = []
        
        for move in moves:
            row, col = move
            
            # 计算移动分数
            score = 0
            
            # 历史启发分数
            score += self.history_table.get((row, col), 0)
            
            # Killer move 分数
            if move in self.killer_moves[depth]:
                score += 1000
            
            # 威胁检测分数
            threat_score = self._get_threat_score(board, row, col, player)
            score += threat_score
            
            # 中心位置分数
            center_distance = abs(row - 7) + abs(col - 7)
            score += (14 - center_distance) * 5
            
            scored_moves.append((score, move))
        
        # 按分数排序
        scored_moves.sort(reverse=True)
        
        return [move for _, move in scored_moves]
    
    def _get_threat_score(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """获取威胁分数"""
        cache_key = (row, col, player, tuple(tuple(row) for row in board))
        
        if cache_key in self.threat_cache:
            return self.threat_cache[cache_key]
        
        threat_score = 0
        
        # 检查是否能形成威胁
        if board[row][col] == EMPTY:
            # 模拟落子
            board[row][col] = player
            
            # 检查是否能获胜
            if self._check_win(board, row, col, player):
                threat_score = 100000
            
            # 检查是否能形成活四
            elif self._has_open_four(board, row, col, player):
                threat_score = 10000
            
            # 检查是否能形成活三
            elif self._has_open_three(board, row, col, player):
                threat_score = 1000
            
            board[row][col] = EMPTY
        
        self.threat_cache[cache_key] = threat_score
        return threat_score
    
    def _check_win(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        for dx, dy in DIRECTIONS:
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
        for dx, dy in DIRECTIONS:
            line = self._get_line_for_pattern(board, row, col, dx, dy, player, 6)
            line_str = ''.join(map(str, line))
            
            # 检查活四模式
            if '022220' in line_str or '011110' in line_str:
                return True
        
        return False
    
    def _has_open_three(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否有活三"""
        for dx, dy in DIRECTIONS:
            line = self._get_line_for_pattern(board, row, col, dx, dy, player, 5)
            line_str = ''.join(map(str, line))
            
            # 检查活三模式
            if '02220' in line_str or '01110' in line_str:
                return True
        
        return False
    
    def _get_line_for_pattern(self, board: List[List[int]], row: int, col: int, 
                             dx: int, dy: int, player: int, length: int) -> List[int]:
        """获取用于模式检测的线"""
        line = []
        
        # 向负方向延伸
        x, y = row, col
        for _ in range(length // 2):
            x -= dx
            y -= dy
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                line.append(board[x][y])
            else:
                line.append(-1)
        
        # 添加当前位置
        line.append(player)
        
        # 向正方向延伸
        x, y = row + dx, col + dy
        for _ in range(length // 2):
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                line.append(board[x][y])
            else:
                line.append(-1)
            x += dx
            y += dy
        
        return line
    
    def update_history_table(self, move: Tuple[int, int], depth: int):
        """更新历史启发表"""
        row, col = move
        self.history_table[(row, col)] += depth * depth
    
    def update_killer_moves(self, move: Tuple[int, int], depth: int):
        """更新killer moves"""
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

class ModernGomokuAI:
    """现代化五子棋AI引擎"""
    
    def __init__(self, max_depth: int = 4, time_limit: float = 5.0):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.evaluator = AdvancedEvaluator()
        self.optimizer = SearchOptimizer()
        
        # 性能统计
        self.nodes_searched = 0
        self.alpha_beta_cutoffs = 0
        self.start_time = 0
        
        # 迭代加深相关
        self.best_move = None
        self.best_score = float('-inf')
        
    def get_best_move(self, board: List[List[int]], player: int) -> SearchResult:
        """获取最佳移动"""
        self.start_time = time.time()
        self.nodes_searched = 0
        self.alpha_beta_cutoffs = 0
        self.best_move = None
        self.best_score = float('-inf')
        
        # 获取候选移动
        candidates = self._get_candidates(board)
        
        if not candidates:
            return SearchResult(
                score=0,
                move=None,
                nodes_searched=0,
                time_elapsed=0,
                depth=0,
                alpha_beta_cutoffs=0
            )
        
        # 迭代加深搜索
        for depth in range(1, self.max_depth + 1):
            # 检查时间限制
            if time.time() - self.start_time > self.time_limit * 0.8:
                break
            
            current_best_move, current_best_score = self._iterative_deepening(
                board, player, depth, candidates
            )
            
            if current_best_move is not None:
                self.best_move = current_best_move
                self.best_score = current_best_score
        
        time_elapsed = time.time() - self.start_time
        
        # 更新移动统计
        if not hasattr(self, 'total_moves'):
            self.total_moves = 0
        if not hasattr(self, 'total_time'):
            self.total_time = 0
        
        self.total_moves += 1
        self.total_time += time_elapsed
        
        return SearchResult(
            score=self.best_score,
            move=self.best_move,
            nodes_searched=self.nodes_searched,
            time_elapsed=time_elapsed,
            depth=self.max_depth,
            alpha_beta_cutoffs=self.alpha_beta_cutoffs
        )
    
    def _iterative_deepening(self, board: List[List[int]], player: int, 
                           depth: int, candidates: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], int]:
        """迭代加深搜索"""
        alpha = float('-inf')
        beta = float('inf')
        
        best_move = None
        best_score = float('-inf')
        
        # 对候选移动进行排序
        sorted_candidates = self.optimizer.get_move_order(board, candidates, player, depth)
        
        for move in sorted_candidates:
            row, col = move
            
            # 检查时间限制
            if time.time() - self.start_time > self.time_limit:
                break
            
            # 模拟落子
            board[row][col] = player
            
            # 执行Alpha-Beta剪枝
            score = -self._alpha_beta(board, 3 - player, depth - 1, -beta, -alpha, False)
            
            # 恢复棋盘
            board[row][col] = EMPTY
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
            
            if alpha >= beta:
                self.alpha_beta_cutoffs += 1
                break
        
        return best_move, best_score
    
    def _alpha_beta(self, board: List[List[int]], player: int, depth: int, 
                   alpha: float, beta: float, is_maximizing: bool) -> float:
        """Alpha-Beta剪枝算法"""
        self.nodes_searched += 1
        
        # 检查时间限制
        if time.time() - self.start_time > self.time_limit:
            return 0
        
        # 终止条件
        if depth == 0:
            return self.evaluator.evaluate_board(board, player)
        
        # 获取候选移动
        candidates = self._get_candidates(board)
        
        if not candidates:
            return 0
        
        # 排序候选移动
        sorted_candidates = self.optimizer.get_move_order(board, candidates, player, depth)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in sorted_candidates:
                row, col = move
                board[row][col] = player
                
                eval_score = self._alpha_beta(board, 3 - player, depth - 1, alpha, beta, False)
                
                board[row][col] = EMPTY
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in sorted_candidates:
                row, col = move
                board[row][col] = player
                
                eval_score = self._alpha_beta(board, 3 - player, depth - 1, alpha, beta, True)
                
                board[row][col] = EMPTY
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    self.alpha_beta_cutoffs += 1
                    break
            
            return min_eval
    
    def _get_candidates(self, board: List[List[int]]) -> List[Tuple[int, int]]:
        """获取候选移动"""
        candidates = []
        
        # 检查是否是空棋盘
        empty_count = sum(row.count(EMPTY) for row in board)
        if empty_count == BOARD_SIZE * BOARD_SIZE:
            # 如果是空棋盘，返回中心位置
            center = BOARD_SIZE // 2
            return [(center, center)]
        
        # 获取所有有邻居的空位置
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY and self._has_neighbor(board, i, j):
                    candidates.append((i, j))
        
        return candidates
    
    def _has_neighbor(self, board: List[List[int]], row: int, col: int) -> bool:
        """检查是否有邻居"""
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = row + di, col + dj
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                    if board[ni][nj] != EMPTY:
                        return True
        
        return False
    
    def get_performance_stats(self) -> Dict[str, any]:
        """获取性能统计"""
        return {
            'nodes_searched': self.nodes_searched,
            'alpha_beta_cutoffs': self.alpha_beta_cutoffs,
            'cutoff_rate': self.alpha_beta_cutoffs / max(1, self.nodes_searched),
            'history_table_size': len(self.optimizer.history_table),
            'threat_cache_size': len(self.optimizer.threat_cache)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要 - 兼容integrated_ai和draw_ai_info的接口"""
        # 计算平均节点数和时间
        avg_nodes = self.nodes_searched / max(1, self.total_moves if hasattr(self, 'total_moves') else 1)
        avg_time = 0.0
        
        if hasattr(self, 'total_time') and self.total_moves > 0:
            avg_time = self.total_time / self.total_moves
        
        return {
            # draw_ai_info期望的嵌套结构
            'ai_stats': {
                'total_moves': getattr(self, 'total_moves', 0),
                'avg_time_per_move': avg_time,
                'avg_nodes_per_move': avg_nodes
            },
            # 其他信息（保持向后兼容）
            'difficulty': f'modern_depth_{self.max_depth}',
            'game_stage': 'unknown',
            'max_depth': self.max_depth,
            'time_limit': self.time_limit,
            'nodes_searched': self.nodes_searched,
            'alpha_beta_cutoffs': self.alpha_beta_cutoffs,
            'cutoff_rate': self.alpha_beta_cutoffs / max(1, self.nodes_searched),
            'architecture': 'Modern Gomoku AI with Advanced Evaluator',
            'search_algorithm': 'Iterative Deepening with Alpha-Beta Pruning'
        }

# 测试函数
def test_ai():
    """测试AI功能"""
    print("正在测试现代化五子棋AI...")
    
    # 创建AI实例
    ai = ModernGomokuAI(max_depth=3, time_limit=2.0)
    
    # 创建测试棋盘
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # 测试开局
    print("测试开局...")
    result = ai.get_best_move(board, BLACK)
    print(f"开局最佳移动: {result.move}, 分数: {result.score}")
    print(f"搜索节点数: {result.nodes_searched}, 用时: {result.time_elapsed:.2f}秒")
    
    # 测试性能统计
    stats = ai.get_performance_stats()
    print(f"性能统计: {stats}")
    
    print("AI测试完成！")

if __name__ == "__main__":
    test_ai()