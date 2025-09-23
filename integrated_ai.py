"""
AI集成模块
将现代化AI系统集成到现有的GomokuGame类中

作者：Claude AI Engineer
日期：2025-09-22
"""

import time
import pygame
import numpy as np
from typing import List, Tuple, Optional, Dict
from modern_ai import ModernGomokuAI, SearchResult
from advanced_evaluator import ComprehensiveEvaluator, ThreatAssessmentSystem
from search_optimizer import SearchOptimizer, SearchStrategy
from neural_evaluator import NeuralNetworkEvaluator, NeuralEvaluatorAdapter, NeuralConfig
from neural_mcts import NeuralMCTSAdapter, NeuralEvaluatorNetworkAdapter, AlphaZeroStyleNetwork

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

class HybridEvaluator:
    """混合评估器，结合传统评估和神经网络评估"""
    
    def __init__(self, traditional_evaluator, neural_adapter, neural_weight: float = 0.4):
        """
        初始化混合评估器
        
        Args:
            traditional_evaluator: 传统评估器
            neural_adapter: 神经网络适配器
            neural_weight: 神经网络评估的权重
        """
        self.traditional_evaluator = traditional_evaluator
        self.neural_adapter = neural_adapter
        self.neural_weight = neural_weight
        self.traditional_weight = 1.0 - neural_weight
        
        # 统计信息
        self.evaluation_count = 0
        self.neural_success_count = 0
        
    def evaluate_board(self, board: List[List[int]], player: int) -> int:
        """评估棋盘状态（兼容modern_ai接口）"""
        return self.comprehensive_evaluate(board, player)
    
    def comprehensive_evaluate(self, board: List[List[int]], player: int) -> int:
        """综合评估棋盘状态"""
        self.evaluation_count += 1
        
        # 传统评估
        traditional_score = self.traditional_evaluator.comprehensive_evaluate(board, player)
        
        # 神经网络评估
        try:
            neural_score = self.neural_adapter.evaluate_board(board, player)
            self.neural_success_count += 1
            
            # 加权组合
            combined_score = (self.traditional_weight * traditional_score + 
                            self.neural_weight * neural_score)
            
            return int(combined_score)
            
        except Exception as e:
            # 如果神经网络评估失败，回退到传统评估
            print(f"神经网络评估失败，使用传统评估: {e}")
            return traditional_score
    
    def evaluate_move(self, board: List[List[int]], row: int, col: int, player: int) -> int:
        """评估移动"""
        if board[row][col] != EMPTY:
            return -float('inf')
        
        # 模拟落子
        board[row][col] = player
        score = self.comprehensive_evaluate(board, player)
        board[row][col] = EMPTY
        
        return score
    
    def get_neural_success_rate(self) -> float:
        """获取神经网络成功率"""
        if self.evaluation_count == 0:
            return 0.0
        return self.neural_success_count / self.evaluation_count

class IntegratedGomokuAI:
    """集成化的五子棋AI系统"""
    
    def __init__(self, ai_difficulty: str = "medium", time_limit: float = 3.0, 
                 use_neural: bool = True, engine_type: str = "minimax",
                 training_mode: bool = False, data_collector = None):
        """
        初始化集成AI系统
        
        Args:
            ai_difficulty: AI难度级别 ("easy", "medium", "hard", "expert", "neural", "neural_mcts")
            time_limit: 时间限制（秒）
            use_neural: 是否使用神经网络评估
            engine_type: AI引擎类型 ("minimax", "neural_mcts")
            training_mode: 是否为训练模式
            data_collector: 训练数据收集器
        """
        self.ai_difficulty = ai_difficulty
        self.time_limit = time_limit
        self.use_neural = use_neural
        self.engine_type = engine_type
        self.training_mode = training_mode
        self.data_collector = data_collector
        
        # 训练模式相关
        self.current_game_id = None
        self.move_start_time = None
        
        # 自动设置引擎类型
        if ai_difficulty == "neural_mcts":
            self.engine_type = "neural_mcts"
            self.use_neural = True
        
        # 根据难度设置参数
        self._setup_ai_parameters()
        
        # 初始化传统评估器
        self.traditional_evaluator = ComprehensiveEvaluator()
        self.threat_assessment = ThreatAssessmentSystem()
        self.search_optimizer = SearchOptimizer(time_limit=self.time_limit)
        
        # 初始化神经网络评估器
        if self.use_neural or ai_difficulty == "neural":
            self.neural_evaluator = self._create_neural_evaluator()
            self.neural_adapter = NeuralEvaluatorAdapter(self.neural_evaluator, weight=0.4)
            self.evaluator = HybridEvaluator(self.traditional_evaluator, self.neural_adapter)
        else:
            self.neural_evaluator = None
            self.neural_adapter = None
            self.evaluator = self.traditional_evaluator
        
        # 初始化AI引擎
        if self.ai_difficulty == "reinforced":
            # 使用强化学习AI引擎
            try:
                from reinforced_ai import create_reinforced_ai_from_best_model
                reinforced_ai = create_reinforced_ai_from_best_model()
                if reinforced_ai:
                    self.ai_engine = reinforced_ai.ai_engine
                    self.neural_evaluator = reinforced_ai.neural_evaluator
                    print("✓ 成功加载强化学习模型")
                else:
                    # 回退到Neural MCTS
                    print("⚠ 未找到训练模型，回退到Neural MCTS")
                    self._create_neural_mcts_engine()
            except ImportError:
                print("⚠ 强化学习模块不可用，回退到Neural MCTS")
                self._create_neural_mcts_engine()
        elif self.engine_type == "neural_mcts" or self.ai_difficulty == "neural_mcts":
            # 使用Neural MCTS引擎
            self._create_neural_mcts_engine()
        else:
            # 使用传统Minimax引擎
            self.ai_engine = ModernGomokuAI(
                max_depth=self.max_depth,
                time_limit=self.time_limit
            )
            
            # 替换AI引擎的评估器
            if hasattr(self.ai_engine, 'evaluator'):
                self.ai_engine.evaluator = self.evaluator
        
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
        
    def _create_neural_evaluator(self) -> NeuralNetworkEvaluator:
        """创建神经网络评估器"""
        # 根据难度配置神经网络
        if self.ai_difficulty == "neural":
            config = NeuralConfig(
                board_size=15,
                input_features=40,
                hidden_layers=[128, 64, 32],
                use_pattern_features=True,
                use_position_features=True,
                use_threat_features=True,
                use_historical_features=True
            )
        elif self.ai_difficulty == "expert":
            config = NeuralConfig(
                board_size=15,
                input_features=32,
                hidden_layers=[64, 32],
                use_pattern_features=True,
                use_position_features=True,
                use_threat_features=True
            )
        else:
            config = NeuralConfig(
                board_size=15,
                input_features=24,
                hidden_layers=[32, 16],
                use_pattern_features=True,
                use_position_features=True
            )
        
        neural_evaluator = NeuralNetworkEvaluator(config)
        
        # 如果有预训练权重，可以在这里加载
        # neural_evaluator.load_weights("pretrained_weights.json")
        
        return neural_evaluator

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
            },
            "neural": {
                "max_depth": 4,
                "time_limit": 4.0,
                "strategy": SearchStrategy.BALANCED
            },
            "neural_mcts": {
                "max_depth": 0,  # MCTS不使用深度限制
                "time_limit": 5.0,
                "strategy": SearchStrategy.BALANCED,
                "mcts_simulations": 1000,
                "c_puct": 1.25
            },
            "reinforced": {
                "max_depth": 0,  # 强化学习不使用深度限制
                "time_limit": 6.0,
                "strategy": SearchStrategy.BALANCED,
                "mcts_simulations": 1200,
                "c_puct": 1.5,
                "use_reinforced_model": True
            }
        }
        
        settings = difficulty_settings.get(self.ai_difficulty, difficulty_settings["medium"])
        self.max_depth = settings["max_depth"]
        self.time_limit = settings["time_limit"]
        self.strategy = settings["strategy"]
        self.mcts_simulations = settings.get("mcts_simulations", 800)
        self.c_puct = settings.get("c_puct", 1.0)
    
    def start_training_game(self, black_player: str = "Neural_MCTS", 
                           white_player: str = "Neural_MCTS", 
                           metadata: Optional[Dict] = None) -> str:
        """
        开始训练游戏
        
        Args:
            black_player: 黑棋玩家标识
            white_player: 白棋玩家标识
            metadata: 元数据
            
        Returns:
            游戏ID
        """
        if self.training_mode and self.data_collector:
            self.current_game_id = self.data_collector.start_game(
                black_player=black_player,
                white_player=white_player,
                metadata=metadata
            )
            return self.current_game_id
        return None
    
    def end_training_game(self, result: Dict, total_time: float = 0) -> bool:
        """
        结束训练游戏
        
        Args:
            result: 游戏结果
            total_time: 总时间
            
        Returns:
            是否成功结束
        """
        if self.training_mode and self.data_collector and self.current_game_id:
            success = self.data_collector.end_game(result, total_time)
            self.current_game_id = None
            return success
        return False
    
    def set_training_mode(self, enabled: bool, data_collector = None):
        """
        设置训练模式
        
        Args:
            enabled: 是否启用训练模式
            data_collector: 数据收集器
        """
        self.training_mode = enabled
        if enabled and data_collector:
            self.data_collector = data_collector
        elif not enabled:
            self.data_collector = None
            self.current_game_id = None
    
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
        self.move_start_time = start_time
        
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
            
            # 训练模式下收集数据
            if self.training_mode and self.data_collector and self.current_game_id:
                self._collect_training_data(board, player, result, start_time)
        
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
    
    def _collect_training_data(self, board: List[List[int]], player: int, 
                              result: SearchResult, start_time: float):
        """
        收集训练数据
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            result: AI搜索结果
            start_time: 开始时间
        """
        try:
            # 转换棋盘为numpy数组
            board_array = np.array(board)
            
            # 获取MCTS概率分布
            mcts_probabilities = {}
            visit_counts = {}
            
            # 从AI引擎获取搜索信息
            if hasattr(self.ai_engine, 'get_search_info'):
                search_info = self.ai_engine.get_search_info()
                mcts_probabilities = search_info.get('move_probabilities', {})
                visit_counts = search_info.get('visit_counts', {})
            elif hasattr(result, 'move_probabilities'):
                mcts_probabilities = getattr(result, 'move_probabilities', {})
                visit_counts = getattr(result, 'visit_counts', {})
            elif hasattr(self.ai_engine, 'mcts') and hasattr(self.ai_engine.mcts, 'get_action_probabilities'):
                # 专门为Neural MCTS适配
                try:
                    action_probs = self.ai_engine.mcts.get_action_probabilities(board_array, player)
                    if action_probs:
                        mcts_probabilities = action_probs
                        # 从访问次数计算
                        if hasattr(self.ai_engine.mcts, 'root') and self.ai_engine.mcts.root:
                            for move, child in self.ai_engine.mcts.root.children.items():
                                visit_counts[move] = child.visit_count
                except Exception as e:
                    print(f"获取MCTS信息失败: {e}")
            
            # 如果没有概率分布，创建简单的分布
            if not mcts_probabilities and result.move:
                mcts_probabilities = {result.move: 1.0}
                visit_counts = {result.move: 1}
            
            # 改进价值估计
            value_estimate = getattr(result, 'value_estimate', result.score / 10000.0)
            
            # 为传统引擎添加价值估计转换
            if self.engine_type != "neural_mcts":
                # 将分数转换为[-1, 1]范围的价值估计
                if abs(result.score) > 100000:  # 必胜局面
                    value_estimate = 1.0 if result.score > 0 else -1.0
                else:
                    # 使用sigmoid函数归一化
                    value_estimate = np.tanh(result.score / 10000.0)
                
                # 从当前玩家角度调整价值
                if player == WHITE:
                    value_estimate = -value_estimate
            
            # 记录移动数据
            thinking_time = time.time() - start_time
            
            success = self.data_collector.record_move(
                position=result.move,
                player=player,
                board_state=board_array,
                mcts_probabilities=mcts_probabilities,
                value_estimate=value_estimate,
                visit_counts=visit_counts,
                search_depth=getattr(result, 'depth', self.max_depth),
                thinking_time=thinking_time
            )
            
            if not success:
                print("警告：训练数据收集失败")
                
            # 添加数据质量检查
            self._validate_training_data(board_array, mcts_probabilities, value_estimate)
                
        except Exception as e:
            print(f"训练数据收集错误: {e}")
    
    def _validate_training_data(self, board_state: np.ndarray, 
                               probabilities: Dict, value_estimate: float):
        """
        验证训练数据质量
        
        Args:
            board_state: 棋盘状态
            probabilities: 概率分布
            value_estimate: 价值估计
        """
        # 检查棋盘状态
        if board_state.shape != (BOARD_SIZE, BOARD_SIZE):
            print(f"警告：棋盘尺寸错误 {board_state.shape}")
            return False
        
        # 检查概率分布
        if probabilities:
            total_prob = sum(probabilities.values())
            if abs(total_prob - 1.0) > 0.1:
                print(f"警告：概率分布和不为1 ({total_prob:.3f})")
            
            # 检查概率值范围
            for pos, prob in probabilities.items():
                if not (0 <= prob <= 1):
                    print(f"警告：概率值超出范围 {pos}: {prob}")
                    return False
                
                # 检查位置有效性
                row, col = pos
                if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                    print(f"警告：位置超出棋盘范围 {pos}")
                    return False
                
                # 检查位置是否为空
                if board_state[row, col] != EMPTY:
                    print(f"警告：位置不为空 {pos}")
                    return False
        
        # 检查价值估计
        if not (-1.5 <= value_estimate <= 1.5):
            print(f"警告：价值估计超出合理范围 {value_estimate}")
        
        return True
    
    def get_training_statistics(self) -> Dict:
        """获取训练统计信息"""
        if self.training_mode and self.data_collector:
            return self.data_collector.get_statistics()
        return {}
    
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
        if self.engine_type == "neural_mcts" or self.ai_difficulty == "neural_mcts":
            # Neural MCTS引擎特性
            base_features = [
                'Neural Monte Carlo Tree Search (MCTS)',
                'PUCT Algorithm with Dynamic c_puct',
                'Neural Network Policy & Value Guidance',
                'Search Tree Reuse',
                'Parallel Search Support',
                'Dirichlet Noise for Exploration',
                'Virtual Loss for Thread Safety',
                'AlphaZero-style Architecture'
            ]
            
            neural_info = {
                'neural_enabled': True,
                'engine_type': 'Neural MCTS',
                'mcts_simulations': self.mcts_simulations,
                'c_puct': self.c_puct
            }
            
            if hasattr(self.ai_engine, 'get_statistics'):
                mcts_stats = self.ai_engine.get_statistics()
                neural_info.update({
                    'mcts_statistics': mcts_stats
                })
        else:
            # 传统Minimax引擎特性
            base_features = [
                'Minimax with Alpha-Beta Pruning',
                'Advanced Pattern Recognition',
                'Threat Assessment System',
                'Heuristic Move Ordering',
                'History Heuristic Table',
                'Killer Move Optimization',
                'Time Control',
                'Iterative Deepening'
            ]
            
            neural_info = {}
            if self.use_neural and self.neural_evaluator:
                base_features.extend([
                    'Neural Network Evaluation',
                    'Hybrid Traditional+Neural Scoring',
                    'Advanced Feature Extraction',
                    'Pattern-Based Neural Features'
                ])
                
                neural_info = {
                    'neural_enabled': True,
                    'neural_model_info': self.neural_evaluator.get_model_info(),
                    'neural_success_rate': getattr(self.evaluator, 'get_neural_success_rate', lambda: 0.0)()
                }
            else:
                neural_info = {'neural_enabled': False}
        
        info = {
            'difficulty': self.ai_difficulty,
            'max_depth': self.max_depth,
            'time_limit': self.time_limit,
            'strategy': self.strategy.name,
            'engine_type': self.engine_type,
            'features': base_features
        }
        
        info.update(neural_info)
        return info

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
    
    def _create_neural_mcts_engine(self):
        """创建Neural MCTS引擎"""
        if self.neural_evaluator:
            network_adapter = NeuralEvaluatorNetworkAdapter(self.neural_evaluator)
        else:
            # 使用默认的dummy网络
            from neural_mcts import AlphaZeroStyleNetwork
            network_adapter = AlphaZeroStyleNetwork()
            
        self.ai_engine = NeuralMCTSAdapter(
            neural_network=network_adapter,
            mcts_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            time_limit=self.time_limit
        )

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