"""
AI集成模块
将现代化AI系统集成到现有的GomokuGame类中

作者：Claude AI Engineer
日期：2025-09-22
"""

import time
import pygame
from typing import List, Tuple, Optional, Dict
from modern_ai import ModernGomokuAI, SearchResult
from advanced_evaluator import ComprehensiveEvaluator, ThreatAssessmentSystem
from search_optimizer import SearchOptimizer, SearchStrategy

# 导入游戏常量
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

class IntegratedGomokuAI:
    """集成化的五子棋AI系统"""
    
    def __init__(self, ai_difficulty: str = "medium", time_limit: float = 3.0):
        """
        初始化集成AI系统
        
        Args:
            ai_difficulty: AI难度级别 ("easy", "medium", "hard", "expert")
            time_limit: 时间限制（秒）
        """
        self.ai_difficulty = ai_difficulty
        self.time_limit = time_limit
        
        # 根据难度设置参数
        self._setup_ai_parameters()
        
        # 初始化AI组件
        self.ai_engine = ModernGomokuAI(
            max_depth=self.max_depth,
            time_limit=self.time_limit
        )
        
        self.evaluator = ComprehensiveEvaluator()
        self.threat_assessment = ThreatAssessmentSystem()
        self.search_optimizer = SearchOptimizer(time_limit=self.time_limit)
        
        # 性能监控
        self.performance_stats = {
            'total_moves': 0,
            'total_time': 0,
            'total_nodes': 0,
            'avg_time_per_move': 0,
            'avg_nodes_per_move': 0,
            'best_moves': []
        }
        
        # 游戏状态
        self.game_stage = "opening"
        self.move_history = []
        
    def _setup_ai_parameters(self):
        """根据难度设置AI参数"""
        difficulty_settings = {
            "easy": {
                "max_depth": 2,
                "time_limit": 1.0,
                "strategy": SearchStrategy.DEFENSIVE
            },
            "medium": {
                "max_depth": 3,
                "time_limit": 2.0,
                "strategy": SearchStrategy.BALANCED
            },
            "hard": {
                "max_depth": 4,
                "time_limit": 3.0,
                "strategy": SearchStrategy.AGGRESSIVE
            },
            "expert": {
                "max_depth": 5,
                "time_limit": 5.0,
                "strategy": SearchStrategy.BALANCED
            }
        }
        
        settings = difficulty_settings.get(self.ai_difficulty, difficulty_settings["medium"])
        self.max_depth = settings["max_depth"]
        self.time_limit = settings["time_limit"]
        self.strategy = settings["strategy"]
    
    def get_ai_move(self, board: List[List[int]], player: int) -> Tuple[int, int]:
        """
        获取AI的移动
        
        Args:
            board: 棋盘状态
            player: AI玩家编号
            
        Returns:
            (row, col): 移动位置
        """
        start_time = time.time()
        
        # 更新游戏阶段
        self._update_game_stage(board)
        
        # 获取候选移动
        candidates = self._get_candidate_moves(board)
        
        if not candidates:
            return None
        
        # 特殊情况处理
        special_move = self._check_special_moves(board, player)
        if special_move:
            return special_move
        
        # 使用AI引擎获取最佳移动
        result = self.ai_engine.get_best_move(board, player)
        
        # 更新性能统计
        self._update_performance_stats(result, start_time)
        
        # 记录移动历史
        if result.move:
            self.move_history.append({
                'move': result.move,
                'score': result.score,
                'time': result.time_elapsed,
                'nodes': result.nodes_searched
            })
        
        return result.move
    
    def _update_game_stage(self, board: List[List[int]]):
        """更新游戏阶段"""
        filled_count = sum(row.count(EMPTY) for row in board)
        empty_count = BOARD_SIZE * BOARD_SIZE - filled_count
        
        if empty_count > BOARD_SIZE * BOARD_SIZE * 0.8:
            self.game_stage = "opening"
        elif empty_count > BOARD_SIZE * BOARD_SIZE * 0.3:
            self.game_stage = "middle"
        else:
            self.game_stage = "endgame"
    
    def _get_candidate_moves(self, board: List[List[int]]) -> List[Tuple[int, int]]:
        """获取候选移动"""
        candidates = []
        
        # 检查是否是空棋盘
        empty_count = sum(row.count(EMPTY) for row in board)
        if empty_count == BOARD_SIZE * BOARD_SIZE:
            # 开局策略
            return [(7, 7)]
        
        # 获取有邻居的空位
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
    
    def _check_special_moves(self, board: List[List[int]], player: int) -> Optional[Tuple[int, int]]:
        """检查特殊情况（必胜、必防等）"""
        # 检查是否有必胜移动
        winning_move = self._find_winning_move(board, player)
        if winning_move:
            return winning_move
        
        # 检查是否需要阻止对手获胜
        opponent = 3 - player
        blocking_move = self._find_winning_move(board, opponent)
        if blocking_move:
            return blocking_move
        
        # 检查关键威胁
        critical_threats = self.threat_assessment.get_critical_threats(board, opponent)
        if critical_threats:
            # 返回威胁评分最高的位置
            best_threat = max(critical_threats, key=lambda t: t.score)
            return best_threat.position
        
        return None
    
    def _find_winning_move(self, board: List[List[int]], player: int) -> Optional[Tuple[int, int]]:
        """寻找必胜移动"""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    # 模拟落子
                    board[i][j] = player
                    if self._check_win(board, i, j, player):
                        board[i][j] = EMPTY
                        return (i, j)
                    board[i][j] = EMPTY
        return None
    
    def _check_win(self, board: List[List[int]], row: int, col: int, player: int) -> bool:
        """检查是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
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
    
    def _update_performance_stats(self, result: SearchResult, start_time: float):
        """更新性能统计"""
        self.performance_stats['total_moves'] += 1
        self.performance_stats['total_time'] += result.time_elapsed
        self.performance_stats['total_nodes'] += result.nodes_searched
        
        # 计算平均值
        total_moves = self.performance_stats['total_moves']
        self.performance_stats['avg_time_per_move'] = (
            self.performance_stats['total_time'] / total_moves
        )
        self.performance_stats['avg_nodes_per_move'] = (
            self.performance_stats['total_nodes'] / total_moves
        )
        
        # 记录最佳移动
        if result.move:
            self.performance_stats['best_moves'].append({
                'move': result.move,
                'score': result.score,
                'time': result.time_elapsed
            })
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        return {
            'difficulty': self.ai_difficulty,
            'game_stage': self.game_stage,
            'total_moves': self.performance_stats['total_moves'],
            'avg_time_per_move': self.performance_stats['avg_time_per_move'],
            'avg_nodes_per_move': self.performance_stats['avg_nodes_per_move'],
            'move_history_count': len(self.move_history)
        }
    
    def get_ai_info(self) -> Dict:
        """获取AI信息"""
        return {
            'difficulty': self.ai_difficulty,
            'max_depth': self.max_depth,
            'time_limit': self.time_limit,
            'strategy': self.strategy.name,
            'engine_type': 'ModernGomokuAI',
            'features': [
                'Minimax with Alpha-Beta Pruning',
                'Advanced Pattern Recognition',
                'Threat Assessment System',
                'Heuristic Move Ordering',
                'History Heuristic Table',
                'Killer Move Optimization',
                'Time Control',
                'Iterative Deepening'
            ]
        }

class EnhancedGomokuGame:
    """增强版五子棋游戏类"""
    
    def __init__(self, ai_difficulty: str = "medium"):
        """初始化增强版游戏"""
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = BLACK  # 1为人类玩家（黑棋），2为电脑玩家（白棋）
        self.game_over = False
        self.winner = 0
        self.human_turn = True
        
        # 集成AI系统
        self.ai_system = IntegratedGomokuAI(ai_difficulty=ai_difficulty)
        
        # UI相关
        self.show_ai_thinking = True
        self.ai_thinking_start_time = 0
        self.last_ai_result = None
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
    def reset_game(self):
        """重置游戏"""
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = BLACK
        self.game_over = False
        self.winner = 0
        self.human_turn = True
        self.last_ai_result = None
        
        # 重置AI系统
        self.ai_system = IntegratedGomokuAI(ai_difficulty=self.ai_system.ai_difficulty)
        
        # 重置性能监控
        self.performance_monitor.reset()
    
    def ai_move(self):
        """AI移动"""
        if self.game_over or self.human_turn:
            return None, None
        
        # 开始性能监控
        self.performance_monitor.start_move()
        self.ai_thinking_start_time = time.time()
        
        # 获取AI移动
        ai_player = WHITE
        move = self.ai_system.get_ai_move(self.board, ai_player)
        
        # 结束性能监控
        self.performance_monitor.end_move()
        
        if move:
            row, col = move
            if self.make_move(row, col):
                self.last_ai_result = {
                    'move': move,
                    'time': time.time() - self.ai_thinking_start_time,
                    'performance': self.ai_system.get_performance_summary()
                }
                return row, col
        
        return None, None
    
    def make_move(self, row: int, col: int) -> bool:
        """落子"""
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            
            # 记录移动
            self.performance_monitor.record_move({
                'player': self.current_player,
                'position': (row, col),
                'time': time.time()
            })
            
            if self.check_winner(row, col):
                self.game_over = True
                self.winner = self.current_player
            else:
                self.current_player = 3 - self.current_player
                self.human_turn = not self.human_turn
            return True
        return False
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """检查落子是否有效"""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col] == EMPTY
        return False
    
    def check_winner(self, row: int, col: int) -> bool:
        """检查是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = self.board[row][col]
        
        for dx, dy in directions:
            count = 1
            
            # 正方向
            x, y = row + dx, col + dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy
            
            # 负方向
            x, y = row - dx, col - dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            
            if count >= 5:
                return True
        
        return False
    
    def get_ai_difficulty(self) -> str:
        """获取AI难度"""
        return self.ai_system.ai_difficulty
    
    def set_ai_difficulty(self, difficulty: str):
        """设置AI难度"""
        self.ai_system = IntegratedGomokuAI(ai_difficulty=difficulty)
    
    def get_ai_info(self) -> Dict:
        """获取AI信息"""
        return self.ai_system.get_ai_info()
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        ai_stats = self.ai_system.get_performance_summary()
        monitor_stats = self.performance_monitor.get_stats()
        
        return {
            'ai_stats': ai_stats,
            'game_stats': monitor_stats
        }

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.move_start_time = 0
        self.moves_data = []
        self.total_moves = 0
        self.total_time = 0
        
    def start_move(self):
        """开始移动计时"""
        self.move_start_time = time.time()
    
    def end_move(self):
        """结束移动计时"""
        if self.move_start_time > 0:
            move_time = time.time() - self.move_start_time
            self.total_moves += 1
            self.total_time += move_time
            self.move_start_time = 0
    
    def record_move(self, move_data: Dict):
        """记录移动数据"""
        self.moves_data.append(move_data)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_time = self.total_time / max(1, self.total_moves)
        
        return {
            'total_moves': self.total_moves,
            'total_time': self.total_time,
            'avg_time_per_move': avg_time,
            'moves_recorded': len(self.moves_data)
        }
    
    def reset(self):
        """重置监控器"""
        self.move_start_time = 0
        self.moves_data = []
        self.total_moves = 0
        self.total_time = 0

# UI增强函数
def draw_ai_info(screen, game: EnhancedGomokuGame, font, small_font):
    """绘制AI信息"""
    # 获取AI信息
    ai_info = game.get_ai_info()
    performance_stats = game.get_performance_stats()
    
    # 绘制AI信息面板
    info_x = 10
    info_y = 10
    line_height = 25
    
    # AI信息标题
    title_text = small_font.render(f"AI难度: {ai_info['difficulty']}", True, (0, 0, 0))
    screen.blit(title_text, (info_x, info_y))
    info_y += line_height
    
    # AI特征
    features_text = small_font.render("引擎特性:", True, (0, 0, 0))
    screen.blit(features_text, (info_x, info_y))
    info_y += line_height
    
    # 显示主要特性
    main_features = ai_info['features'][:3]  # 只显示前3个主要特性
    for feature in main_features:
        feature_text = small_font.render(f"  • {feature}", True, (0, 0, 0))
        screen.blit(feature_text, (info_x, info_y))
        info_y += line_height
    
    # 性能统计
    if performance_stats['ai_stats']['total_moves'] > 0:
        info_y += 10
        perf_text = small_font.render("性能统计:", True, (0, 0, 0))
        screen.blit(perf_text, (info_x, info_y))
        info_y += line_height
        
        avg_time = performance_stats['ai_stats']['avg_time_per_move']
        avg_nodes = performance_stats['ai_stats']['avg_nodes_per_move']
        
        time_text = small_font.render(f"  平均时间: {avg_time:.2f}秒", True, (0, 0, 0))
        screen.blit(time_text, (info_x, info_y))
        info_y += line_height
        
        nodes_text = small_font.render(f"  平均节点: {avg_nodes:.0f}", True, (0, 0, 0))
        screen.blit(nodes_text, (info_x, info_y))
        info_y += line_height

def draw_ai_thinking(screen, game: EnhancedGomokuGame, font):
    """绘制AI思考动画"""
    if not game.human_turn and not game.game_over and game.show_ai_thinking:
        # 计算思考时间
        thinking_time = time.time() - game.ai_thinking_start_time
        
        # 绘制思考动画
        thinking_text = font.render("AI思考中...", True, (255, 0, 0))
        text_rect = thinking_text.get_rect(center=(screen.get_width() // 2, 50))
        screen.blit(thinking_text, text_rect)
        
        # 绘制思考时间
        time_text = font.render(f"思考时间: {thinking_time:.1f}秒", True, (255, 0, 0))
        time_rect = time_text.get_rect(center=(screen.get_width() // 2, 80))
        screen.blit(time_text, time_rect)
        
        # 绘制搜索深度指示器
        depth_text = font.render(f"搜索深度: {game.ai_system.max_depth}", True, (255, 0, 0))
        depth_rect = depth_text.get_rect(center=(screen.get_width() // 2, 110))
        screen.blit(depth_text, depth_rect)

def draw_last_ai_move(screen, game: EnhancedGomokuGame, small_font):
    """绘制最后AI移动信息"""
    if game.last_ai_result:
        # 绘制最后移动信息
        move = game.last_ai_result['move']
        move_time = game.last_ai_result['time']
        
        info_text = small_font.render(
            f"最后AI移动: ({move[0]}, {move[1]}) 用时: {move_time:.2f}秒",
            True, (0, 0, 255)
        )
        
        info_rect = info_text.get_rect(center=(screen.get_width() // 2, screen.get_height() - 30))
        screen.blit(info_text, info_rect)

# 测试函数
def test_integrated_ai():
    """测试集成AI系统"""
    print("正在测试集成AI系统...")
    
    # 创建增强版游戏
    game = EnhancedGomokuAI(ai_difficulty="medium")
    
    # 创建测试棋盘
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # 设置一些测试棋子
    board[7][7] = BLACK
    board[7][8] = WHITE
    board[8][7] = WHITE
    
    # 测试AI移动
    print("测试AI移动...")
    ai_move = game.get_ai_move(board, WHITE)
    print(f"AI移动: {ai_move}")
    
    # 获取AI信息
    ai_info = game.get_ai_info()
    print(f"AI信息: {ai_info}")
    
    # 获取性能摘要
    performance = game.get_performance_summary()
    print(f"性能摘要: {performance}")
    
    print("集成AI系统测试完成！")

if __name__ == "__main__":
    test_integrated_ai()