"""
高级评估函数模块
实现精确的棋型识别、威胁等级系统和全局战略评估

作者：Claude AI Engineer
日期：2025-09-22
"""

from typing import List, Tuple, Dict, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import math

# 游戏常量
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

class ThreatLevel(Enum):
    """威胁等级枚举"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    WINNING = 5

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
    DOUBLE_FOUR = 10
    FORK = 11

@dataclass
class PatternInfo:
    """棋型信息"""
    pattern_type: PatternType
    positions: List[Tuple[int, int]]
    score: int
    threat_level: ThreatLevel

@dataclass
class ThreatInfo:
    """威胁信息"""
    position: Tuple[int, int]
    level: ThreatLevel
    score: int
    patterns: List[PatternType]

class AdvancedPatternRecognition:
    """高级棋型识别系统"""
    
    def __init__(self):
        # 棋型定义字典
        self.patterns = self._create_pattern_definitions()
        
        # 棋型分数映射
        self.pattern_scores = {
            PatternType.FIVE: 100000,
            PatternType.OPEN_FOUR: 10000,
            PatternType.BLOCKED_FOUR: 5000,
            PatternType.DOUBLE_FOUR: 80000,
            PatternType.DOUBLE_THREE: 5000,
            PatternType.OPEN_THREE: 1000,
            PatternType.BLOCKED_THREE: 100,
            PatternType.OPEN_TWO: 200,
            PatternType.BLOCKED_TWO: 50,
            PatternType.SINGLE: 10,
            PatternType.FORK: 15000,
        }
        
        # 威胁等级映射
        self.threat_mapping = {
            PatternType.FIVE: ThreatLevel.WINNING,
            PatternType.OPEN_FOUR: ThreatLevel.CRITICAL,
            PatternType.BLOCKED_FOUR: ThreatLevel.HIGH,
            PatternType.DOUBLE_FOUR: ThreatLevel.WINNING,
            PatternType.DOUBLE_THREE: ThreatLevel.HIGH,
            PatternType.OPEN_THREE: ThreatLevel.MEDIUM,
            PatternType.BLOCKED_THREE: ThreatLevel.LOW,
            PatternType.OPEN_TWO: ThreatLevel.LOW,
            PatternType.BLOCKED_TWO: ThreatLevel.LOW,
            PatternType.SINGLE: ThreatLevel.NONE,
            PatternType.FORK: ThreatLevel.CRITICAL,
        }
        
    def _create_pattern_definitions(self) -> Dict[str, PatternType]:
        """创建棋型定义字典"""
        patterns = {}
        
        # 五连
        patterns['22222'] = PatternType.FIVE
        patterns['11111'] = PatternType.FIVE
        
        # 活四
        patterns['022220'] = PatternType.OPEN_FOUR
        patterns['011110'] = PatternType.OPEN_FOUR
        
        # 冲四
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
        
        # 活三
        patterns['02220'] = PatternType.OPEN_THREE
        patterns['01110'] = PatternType.OPEN_THREE
        patterns['002220'] = PatternType.OPEN_THREE
        patterns['011100'] = PatternType.OPEN_THREE
        
        # 眠三
        patterns['0222'] = PatternType.BLOCKED_THREE
        patterns['2220'] = PatternType.BLOCKED_THREE
        patterns['0111'] = PatternType.BLOCKED_THREE
        patterns['1110'] = PatternType.BLOCKED_THREE
        patterns['2022'] = PatternType.BLOCKED_THREE
        patterns['2202'] = PatternType.BLOCKED_THREE
        patterns['02022'] = PatternType.BLOCKED_THREE
        patterns['02202'] = PatternType.BLOCKED_THREE
        
        # 活二
        patterns['0220'] = PatternType.OPEN_TWO
        patterns['0110'] = PatternType.OPEN_TWO
        patterns['00220'] = PatternType.OPEN_TWO
        patterns['00110'] = PatternType.OPEN_TWO
        
        # 眠二
        patterns['022'] = PatternType.BLOCKED_TWO
        patterns['220'] = PatternType.BLOCKED_TWO
        patterns['011'] = PatternType.BLOCKED_TWO
        patterns['110'] = PatternType.BLOCKED_TWO
        
        return patterns
    
    def identify_patterns(self, board: List[List[int]], player: int) -> List[PatternInfo]:
        """识别所有棋型"""
        patterns = []
        visited_positions = set()
        
        # 在所有方向上寻找棋型
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if board[i][j] == player and (i, j) not in visited_positions:
                        pattern_info = self._identify_pattern_at_position(
                            board, i, j, dx, dy, player
                        )
                        if pattern_info:
                            patterns.append(pattern_info)
                            # 标记已访问的位置
                            for pos in pattern_info.positions:
                                visited_positions.add(pos)
        
        # 识别特殊棋型（双三、双四、分叉等）
        special_patterns = self._identify_special_patterns(board, player, patterns)
        patterns.extend(special_patterns)
        
        return patterns
    
    def _identify_pattern_at_position(self, board: List[List[int]], row: int, col: int, 
                                   dx: int, dy: int, player: int) -> PatternInfo:
        """在指定位置识别棋型"""
        # 获取该方向的线
        line = self._get_line(board, row, col, dx, dy, 8)
        line_str = ''.join(map(str, line))
        
        # 查找匹配的棋型
        for pattern_str, pattern_type in self.patterns.items():
            if pattern_str in line_str:
                # 获取棋型位置
                positions = self._get_pattern_positions(board, row, col, dx, dy, pattern_str, player)
                
                score = self.pattern_scores.get(pattern_type, 0)
                threat_level = self.threat_mapping.get(pattern_type, ThreatLevel.NONE)
                
                return PatternInfo(
                    pattern_type=pattern_type,
                    positions=positions,
                    score=score,
                    threat_level=threat_level
                )
        
        return None
    
    def _get_line(self, board: List[List[int]], row: int, col: int, dx: int, dy: int, length: int) -> List[int]:
        """获取指定方向的线"""
        line = []
        
        # 向两个方向延伸
        for direction in [-1, 1]:
            x, y = row, col
            for _ in range(length // 2):
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
        
        return line
    
    def _get_pattern_positions(self, board: List[List[int]], row: int, col: int, 
                              dx: int, dy: int, pattern_str: str, player: int) -> List[Tuple[int, int]]:
        """获取棋型位置"""
        positions = []
        
        # 从当前位置开始寻找模式
        start_index = 4  # 假设当前位置在中间
        
        # 在线中查找模式
        line = self._get_line(board, row, col, dx, dy, 8)
        line_str = ''.join(map(str, line))
        
        pattern_index = line_str.find(pattern_str)
        if pattern_index != -1:
            # 计算实际位置
            for i, char in enumerate(pattern_str):
                if char == str(player):
                    # 计算在棋盘上的位置
                    actual_index = pattern_index + i
                    offset = actual_index - start_index
                    
                    if offset == 0:
                        positions.append((row, col))
                    else:
                        new_row = row + offset * dx
                        new_col = col + offset * dy
                        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                            positions.append((new_row, new_col))
        
        return positions
    
    def _identify_special_patterns(self, board: List[List[int]], player: int, 
                                 base_patterns: List[PatternInfo]) -> List[PatternInfo]:
        """识别特殊棋型（双三、双四、分叉等）"""
        special_patterns = []
        
        # 统计各类型棋型数量
        pattern_counts = defaultdict(int)
        pattern_positions = defaultdict(list)
        
        for pattern in base_patterns:
            pattern_counts[pattern.pattern_type] += 1
            pattern_positions[pattern.pattern_type].extend(pattern.positions)
        
        # 检测双活三
        if pattern_counts[PatternType.OPEN_THREE] >= 2:
            special_patterns.append(PatternInfo(
                pattern_type=PatternType.DOUBLE_THREE,
                positions=list(set(pattern_positions[PatternType.OPEN_THREE])),
                score=self.pattern_scores[PatternType.DOUBLE_THREE],
                threat_level=self.threat_mapping[PatternType.DOUBLE_THREE]
            ))
        
        # 检测双活四
        if pattern_counts[PatternType.OPEN_FOUR] >= 2:
            special_patterns.append(PatternInfo(
                pattern_type=PatternType.DOUBLE_FOUR,
                positions=list(set(pattern_positions[PatternType.OPEN_FOUR])),
                score=self.pattern_scores[PatternType.DOUBLE_FOUR],
                threat_level=self.threat_mapping[PatternType.DOUBLE_FOUR]
            ))
        
        # 检测分叉（多个方向的威胁）
        threat_positions = set()
        for pattern in base_patterns:
            if pattern.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                threat_positions.update(pattern.positions)
        
        if len(threat_positions) >= 2:
            special_patterns.append(PatternInfo(
                pattern_type=PatternType.FORK,
                positions=list(threat_positions),
                score=self.pattern_scores[PatternType.FORK],
                threat_level=self.threat_mapping[PatternType.FORK]
            ))
        
        return special_patterns

class ThreatAssessmentSystem:
    """威胁评估系统"""
    
    def __init__(self):
        self.pattern_recognizer = AdvancedPatternRecognition()
        
    def assess_threats(self, board: List[List[int]], player: int) -> List[ThreatInfo]:
        """评估所有威胁"""
        threats = []
        
        # 识别棋型
        patterns = self.pattern_recognizer.identify_patterns(board, player)
        
        # 按威胁等级分组
        threat_groups = defaultdict(list)
        for pattern in patterns:
            threat_groups[pattern.threat_level].append(pattern)
        
        # 创建威胁信息
        for threat_level, pattern_list in threat_groups.items():
            if threat_level != ThreatLevel.NONE:
                # 合并相同威胁等级的模式
                all_positions = set()
                all_pattern_types = set()
                total_score = 0
                
                for pattern in pattern_list:
                    all_positions.update(pattern.positions)
                    all_pattern_types.add(pattern.pattern_type)
                    total_score += pattern.score
                
                # 为每个位置创建威胁信息
                for position in all_positions:
                    threats.append(ThreatInfo(
                        position=position,
                        level=threat_level,
                        score=total_score,
                        patterns=list(all_pattern_types)
                    ))
        
        # 按威胁等级排序
        threats.sort(key=lambda x: x.level.value, reverse=True)
        
        return threats
    
    def get_critical_threats(self, board: List[List[int]], player: int) -> List[ThreatInfo]:
        """获取关键威胁"""
        threats = self.assess_threats(board, player)
        return [t for t in threats if t.level in [ThreatLevel.CRITICAL, ThreatLevel.WINNING]]
    
    def get_defensive_moves(self, board: List[List[int]], opponent: int) -> List[Tuple[int, int]]:
        """获取防守移动"""
        defensive_moves = []
        
        # 获取对手的威胁
        opponent_threats = self.assess_threats(board, opponent)
        
        # 优先处理关键威胁
        for threat in opponent_threats:
            if threat.level in [ThreatLevel.CRITICAL, ThreatLevel.WINNING]:
                # 找到该威胁的关键防守位置
                defensive_positions = self._find_defensive_positions(board, threat)
                defensive_moves.extend(defensive_positions)
        
        return list(set(defensive_moves))
    
    def _find_defensive_positions(self, board: List[List[int]], threat: ThreatInfo) -> List[Tuple[int, int]]:
        """找到防守位置"""
        defensive_positions = []
        
        # 根据威胁类型确定防守策略
        if threat.level == ThreatLevel.WINNING:
            # 必须立即阻止
            for pattern_type in threat.patterns:
                if pattern_type == PatternType.FIVE:
                    # 找到能阻止五连的位置
                    defensive_positions.extend(self._find_five_defense(board, threat.position))
        
        elif threat.level == ThreatLevel.CRITICAL:
            # 高优先级防守
            for pattern_type in threat.patterns:
                if pattern_type == PatternType.OPEN_FOUR:
                    defensive_positions.extend(self._find_open_four_defense(board, threat.position))
                elif pattern_type == PatternType.FORK:
                    defensive_positions.extend(self._find_fork_defense(board, threat.position))
        
        return defensive_positions
    
    def _find_five_defense(self, board: List[List[int]], position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """找到阻止五连的位置"""
        defensive_positions = []
        row, col = position
        
        # 检查所有方向
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            # 找到五连的空位
            empty_positions = []
            x, y = row, col
            
            # 向正方向
            for _ in range(5):
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    if board[x][y] == EMPTY:
                        empty_positions.append((x, y))
                x += dx
                y += dy
            
            # 向负方向
            x, y = row - dx, col - dy
            for _ in range(5):
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    if board[x][y] == EMPTY:
                        empty_positions.append((x, y))
                x -= dx
                y -= dy
            
            defensive_positions.extend(empty_positions)
        
        return defensive_positions
    
    def _find_open_four_defense(self, board: List[List[int]], position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """找到阻止活四的位置"""
        # 简化版本：返回威胁位置周围的空位
        defensive_positions = []
        row, col = position
        
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = row + di, col + dj
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                    if board[ni][nj] == EMPTY:
                        defensive_positions.append((ni, nj))
        
        return defensive_positions
    
    def _find_fork_defense(self, board: List[List[int]], position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """找到阻止分叉的位置"""
        # 简化版本：返回威胁位置
        defensive_positions = []
        row, col = position
        
        if board[row][col] == EMPTY:
            defensive_positions.append((row, col))
        
        return defensive_positions

class StrategicEvaluator:
    """战略评估器"""
    
    def __init__(self):
        # 位置权重矩阵
        self.position_weights = self._create_position_weights()
        
        # 方向权重
        self.direction_weights = {
            (0, 1): 1.0,    # 水平
            (1, 0): 1.0,    # 垂直
            (1, 1): 1.2,    # 主对角线
            (1, -1): 1.2,   # 副对角线
        }
        
        # 阶段性权重
        self.stage_weights = {
            'opening': 1.0,
            'middle': 1.2,
            'endgame': 1.5,
        }
        
    def _create_position_weights(self) -> List[List[int]]:
        """创建位置权重矩阵"""
        weights = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        center = BOARD_SIZE // 2
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                # 计算到中心的曼哈顿距离
                distance = abs(i - center) + abs(j - center)
                
                # 中心位置权重最高
                if distance <= 2:
                    weights[i][j] = 20
                elif distance <= 4:
                    weights[i][j] = 15
                elif distance <= 6:
                    weights[i][j] = 10
                else:
                    weights[i][j] = 5
        
        return weights
    
    def evaluate_strategic_position(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估战略位置"""
        score = 0
        
        # 位置权重
        score += self.position_weights[row][col] * 10
        
        # 控制力评估
        control_score = self._evaluate_control(board, row, col, player)
        score += control_score
        
        # 连通性评估
        connectivity_score = self._evaluate_connectivity(board, row, col, player)
        score += connectivity_score
        
        # 边缘评估
        edge_score = self._evaluate_edge_position(row, col)
        score += edge_score
        
        return score
    
    def _evaluate_control(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估控制力"""
        control_score = 0
        
        # 检查周围的控制情况
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = row + di, col + dj
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                    if board[ni][nj] == player:
                        # 自己的棋子
                        distance = abs(di) + abs(dj)
                        control_score += (5 - distance) * 5
                    elif board[ni][nj] == 3 - player:
                        # 对手的棋子
                        distance = abs(di) + abs(dj)
                        control_score -= (5 - distance) * 3
        
        return control_score
    
    def _evaluate_connectivity(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估连通性"""
        connectivity_score = 0
        
        # 检查与己方棋子的连通性
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            connected_count = 0
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                if board[x][y] == player:
                    connected_count += 1
                else:
                    break
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                if board[x][y] == player:
                    connected_count += 1
                else:
                    break
                x -= dx
                y -= dy
            
            # 根据连通性评分
            if connected_count >= 2:
                connectivity_score += connected_count * 10
        
        return connectivity_score
    
    def _evaluate_edge_position(self, row: int, col: int) -> int:
        """评估边缘位置"""
        edge_score = 0
        
        # 边缘位置惩罚
        if row <= 1 or row >= BOARD_SIZE - 2:
            edge_score -= 10
        if col <= 1 or col >= BOARD_SIZE - 2:
            edge_score -= 10
        
        # 角落位置惩罚
        if (row <= 1 and col <= 1) or (row <= 1 and col >= BOARD_SIZE - 2) or \
           (row >= BOARD_SIZE - 2 and col <= 1) or (row >= BOARD_SIZE - 2 and col >= BOARD_SIZE - 2):
            edge_score -= 20
        
        return edge_score
    
    def get_game_stage(self, board: List[List[int]]) -> str:
        """获取游戏阶段"""
        filled_positions = sum(row.count(EMPTY) for row in board)
        empty_positions = BOARD_SIZE * BOARD_SIZE - filled_positions
        
        if empty_positions > BOARD_SIZE * BOARD_SIZE * 0.8:
            return 'opening'
        elif empty_positions > BOARD_SIZE * BOARD_SIZE * 0.3:
            return 'middle'
        else:
            return 'endgame'

class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        self.pattern_recognizer = AdvancedPatternRecognition()
        self.threat_assessment = ThreatAssessmentSystem()
        self.strategic_evaluator = StrategicEvaluator()
        
    def comprehensive_evaluate(self, board: List[List[int]], player: int) -> int:
        """综合评估"""
        total_score = 0
        
        # 1. 棋型评估
        patterns = self.pattern_recognizer.identify_patterns(board, player)
        pattern_score = sum(pattern.score for pattern in patterns)
        total_score += pattern_score
        
        # 2. 威胁评估
        threats = self.threat_assessment.assess_threats(board, player)
        threat_score = sum(threat.score for threat in threats)
        total_score += threat_score
        
        # 3. 战略评估
        game_stage = self.strategic_evaluator.get_game_stage(board)
        stage_multiplier = self.strategic_evaluator.stage_weights.get(game_stage, 1.0)
        
        # 对所有空位进行战略评估
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    strategic_score = self.strategic_evaluator.evaluate_strategic_position(
                        board, i, j, player
                    )
                    total_score += strategic_score * 0.1  # 降低战略评估的权重
        
        # 应用阶段权重
        total_score *= stage_multiplier
        
        return total_score
    
    def evaluate_move(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估移动"""
        if board[row][col] != EMPTY:
            return -float('inf')
        
        # 模拟落子
        board[row][col] = player
        score = self.comprehensive_evaluate(board, player)
        board[row][col] = EMPTY
        
        return score

# 测试函数
def test_advanced_evaluation():
    """测试高级评估函数"""
    print("正在测试高级评估函数...")
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator()
    
    # 创建测试棋盘
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # 设置测试棋局
    board[7][7] = BLACK
    board[7][8] = WHITE
    board[8][7] = WHITE
    board[8][8] = BLACK
    
    # 测试棋型识别
    patterns = evaluator.pattern_recognizer.identify_patterns(board, BLACK)
    print(f"识别到 {len(patterns)} 个棋型")
    
    # 测试威胁评估
    threats = evaluator.threat_assessment.assess_threats(board, BLACK)
    print(f"识别到 {len(threats)} 个威胁")
    
    # 测试综合评估
    score = evaluator.comprehensive_evaluate(board, BLACK)
    print(f"综合评估分数: {score}")
    
    # 测试移动评估
    move_score = evaluator.evaluate_move(board, 7, 9, BLACK)
    print(f"移动 (7,9) 评估分数: {move_score}")
    
    print("高级评估函数测试完成！")

if __name__ == "__main__":
    test_advanced_evaluation()