"""
神经网络增强的蒙特卡洛树搜索（Neural MCTS）
基于AlphaZero算法的MCTS实现

作者：Claude AI Engineer
日期：2025-09-22
"""

import math
import random
import numpy as np
import threading
import concurrent.futures
from threading import Lock
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MCTSNode:
    """MCTS节点"""
    state: np.ndarray  # 棋盘状态
    parent: Optional['MCTSNode'] = None
    children: Dict[Tuple[int, int], 'MCTSNode'] = None
    
    # MCTS统计信息
    visit_count: int = 0
    total_value: float = 0.0
    prior_probability: float = 0.0
    
    # 神经网络预测
    policy_probs: Dict[Tuple[int, int], float] = None
    value_estimate: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.policy_probs is None:
            self.policy_probs = {}
    
    @property
    def is_expanded(self) -> bool:
        """判断节点是否已展开"""
        return len(self.children) > 0
    
    @property
    def average_value(self) -> float:
        """平均价值"""
        return self.total_value / max(1, self.visit_count)

class NeuralMCTS:
    """神经网络增强的MCTS"""
    
    def __init__(self, neural_network, c_puct: float = 1.0, num_simulations: int = 800, 
                 add_dirichlet_noise: bool = False, dirichlet_alpha: float = 0.3,
                 virtual_loss: int = 3, cpuct_init: float = 1.25, cpuct_base: float = 19652,
                 enable_parallel: bool = False, num_threads: int = 4):
        """
        初始化Neural MCTS
        
        Args:
            neural_network: 神经网络模型（策略+价值网络）
            c_puct: PUCT公式中的探索常数（类似AlphaZero的cpuct）
            num_simulations: 模拟次数
            add_dirichlet_noise: 是否在根节点添加Dirichlet噪声
            dirichlet_alpha: Dirichlet噪声参数
            virtual_loss: 虚拟损失（用于并行搜索）
            cpuct_init: c_puct初始值
            cpuct_base: c_puct动态调整参数
            enable_parallel: 是否启用并行搜索
            num_threads: 并行线程数
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.virtual_loss = virtual_loss
        self.cpuct_init = cpuct_init
        self.cpuct_base = cpuct_base
        self.enable_parallel = enable_parallel
        self.num_threads = num_threads
        
        # 统计信息
        self.total_simulations = 0
        self.cache_hits = 0
        
        # 缓存机制
        self.evaluation_cache = {}
        
        # 搜索控制
        self.max_cache_size = 10000
        self.clear_cache_frequency = 1000
        
        # 并行搜索用的锁
        self.cache_lock = Lock()
        self.stats_lock = Lock()
        
        # 搜索树重用
        self.tree_reuse_enabled = True
        self.previous_root = None
        
        # 训练数据收集
        self.data_collection_enabled = False
        self.move_probabilities = {}
        self.visit_counts = {}
        self.value_estimate = 0.0
    
    def search(self, root_state: np.ndarray, player: int, 
               previous_action: Optional[Tuple[int, int]] = None) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        执行MCTS搜索（支持搜索树重用和并行搜索）
        
        Args:
            root_state: 根节点状态
            player: 当前玩家
            previous_action: 上一步动作（用于搜索树重用）
            
        Returns:
            (action_probs, root_value): 动作概率分布和根节点价值
        """
        # 创建或重用根节点
        root = self._get_or_create_root(root_state, previous_action)
        
        # 评估根节点
        if not root.is_expanded:
            self._evaluate_node(root, player)
        
        # 执行模拟
        if self.enable_parallel:
            self._parallel_search(root, player)
        else:
            self._sequential_search(root, player)
        
        # 计算访问概率分布
        action_probs = self._get_action_probabilities(root)
        
        # 收集训练数据
        if self.data_collection_enabled:
            self.move_probabilities = action_probs.copy()
            self.visit_counts = {action: child.visit_count for action, child in root.children.items()}
            self.value_estimate = root.average_value
        
        # 保存根节点用于下次重用
        if self.tree_reuse_enabled:
            self.previous_root = root
        
        return action_probs, root.average_value
    
    def _get_or_create_root(self, root_state: np.ndarray, 
                           previous_action: Optional[Tuple[int, int]]) -> MCTSNode:
        """
        获取或创建根节点（支持搜索树重用）
        
        Args:
            root_state: 根节点状态
            previous_action: 上一步动作
            
        Returns:
            根节点
        """
        # 如果不启用搜索树重用或没有上一次的根节点，创建新节点
        if not self.tree_reuse_enabled or self.previous_root is None or previous_action is None:
            return MCTSNode(state=root_state.copy())
        
        # 尝试从上一次的搜索树中重用
        if previous_action in self.previous_root.children:
            reused_node = self.previous_root.children[previous_action]
            reused_node.parent = None  # 设置为新的根节点
            print(f"搜索树重用: 节点访问次数 {reused_node.visit_count}")
            return reused_node
        
        # 如果找不到对应的子节点，创建新节点
        return MCTSNode(state=root_state.copy())
    
    def _sequential_search(self, root: MCTSNode, player: int):
        """顺序搜索"""
        for _ in range(self.num_simulations):
            self._simulate(root, player)
            self.total_simulations += 1
    
    def _parallel_search(self, root: MCTSNode, player: int):
        """
        并行搜索
        
        Args:
            root: 根节点
            player: 当前玩家
        """
        # 计算每个线程的模拟次数
        sims_per_thread = self.num_simulations // self.num_threads
        remaining_sims = self.num_simulations % self.num_threads
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            
            for i in range(self.num_threads):
                # 给最后一个线程分配剩余的模拟次数
                thread_sims = sims_per_thread + (remaining_sims if i == self.num_threads - 1 else 0)
                future = executor.submit(self._thread_search, root, player, thread_sims)
                futures.append(future)
            
            # 等待所有线程完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"并行搜索线程错误: {e}")
    
    def _thread_search(self, root: MCTSNode, player: int, num_sims: int):
        """
        单线程搜索
        
        Args:
            root: 根节点
            player: 当前玩家
            num_sims: 该线程的模拟次数
        """
        for _ in range(num_sims):
            self._simulate_with_virtual_loss(root, player)
            
            with self.stats_lock:
                self.total_simulations += 1
    
    def _simulate_with_virtual_loss(self, node: MCTSNode, player: int) -> float:
        """
        带虚拟损失的模拟（用于并行搜索）
        
        Args:
            node: 当前节点
            player: 当前玩家
            
        Returns:
            价值估计
        """
        # 在选择路径上应用虚拟损失
        path = []
        current_node = node
        current_player = player
        
        # 选择阶段：向下选择直到叶子节点
        while current_node.is_expanded:
            action, child_node = self._select_child(current_node)
            path.append((current_node, action, child_node))
            
            # 应用虚拟损失
            child_node.visit_count += self.virtual_loss
            child_node.total_value -= self.virtual_loss  # 负值表示损失
            
            current_node = child_node
            current_player = 3 - current_player
        
        # 扩展和评估叶子节点
        value = self._expand_and_evaluate(current_node, current_player)
        
        # 回传阶段：更新路径上的所有节点
        for node_in_path, action, child_node in reversed(path):
            # 移除虚拟损失
            child_node.visit_count -= self.virtual_loss
            child_node.total_value += self.virtual_loss
            
            # 应用真实更新
            child_node.visit_count += 1
            child_node.total_value += value
            
            # 下一层的价值需要取反
            value = -value
        
        # 更新根节点
        node.visit_count += 1
        node.total_value += value
        
        return value
    
    def _simulate(self, node: MCTSNode, player: int) -> float:
        """
        执行一次模拟
        
        Args:
            node: 当前节点
            player: 当前玩家
            
        Returns:
            价值估计
        """
        # 1. 选择阶段：选择最有希望的子节点
        if node.is_expanded:
            action, next_node = self._select_child(node)
            value = -self._simulate(next_node, 3 - player)
        else:
            # 2. 扩展阶段：展开节点
            value = self._expand_and_evaluate(node, player)
        
        # 3. 回传阶段：更新节点统计信息
        node.visit_count += 1
        node.total_value += value
        
        return value
    
    def _get_dynamic_cpuct(self, node: MCTSNode) -> float:
        """
        动态计算c_puct值（根据AlphaZero论文）
        
        c_puct = cpuct_init + log((1 + N + cpuct_base) / cpuct_base)
        其中N是父节点的访问次数
        
        Args:
            node: 父节点
            
        Returns:
            动态c_puct值
        """
        n_visits = node.visit_count
        dynamic_cpuct = self.cpuct_init + math.log((1 + n_visits + self.cpuct_base) / self.cpuct_base)
        return dynamic_cpuct
    
    def _select_child(self, node: MCTSNode) -> Tuple[Tuple[int, int], MCTSNode]:
        """
        选择最优子节点（基于PUCT算法）
        
        PUCT公式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        其中:
        - Q(s,a): 平均价值（利用项）
        - P(s,a): 先验概率（神经网络策略输出）
        - c_puct: 探索常数
        - N(s): 父节点访问次数
        - N(s,a): 子节点访问次数
        
        Args:
            node: 父节点
            
        Returns:
            (action, child_node): 最优动作和子节点
        """
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        # PUCT公式中的父节点访问次数项
        sqrt_parent_visits = math.sqrt(node.visit_count)
        
        # 动态c_puct值
        current_cpuct = self._get_dynamic_cpuct(node)
        
        for action, child in node.children.items():
            # Q(s,a) - 平均价值（利用项）
            q_value = child.average_value
            
            # P(s,a) - 先验概率（神经网络策略输出）
            prior_prob = child.prior_probability
            
            # U(s,a) - 探索项：c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            u_value = (current_cpuct * prior_prob * sqrt_parent_visits / 
                      (1 + child.visit_count))
            
            # PUCT分数 = Q(s,a) + U(s,a)
            puct_score = q_value + u_value
            
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _expand_and_evaluate(self, node: MCTSNode, player: int) -> float:
        """
        展开节点并评估
        
        Args:
            node: 要展开的节点
            player: 当前玩家
            
        Returns:
            价值估计
        """
        # 评估当前节点
        self._evaluate_node(node, player)
        
        # 创建子节点
        legal_actions = self._get_legal_actions(node.state)
        
        for action in legal_actions:
            # 创建新状态
            new_state = node.state.copy()
            row, col = action
            new_state[row, col] = player
            
            # 创建子节点
            child = MCTSNode(
                state=new_state,
                parent=node,
                prior_probability=node.policy_probs.get(action, 0.0)
            )
            
            node.children[action] = child
        
        return node.value_estimate
    
    def _evaluate_node(self, node: MCTSNode, player: int):
        """
        使用神经网络评估节点（线程安全）
        
        Args:
            node: 要评估的节点
            player: 当前玩家
        """
        # 检查缓存（线程安全）
        state_key = self._get_state_key(node.state, player)
        
        with self.cache_lock:
            if state_key in self.evaluation_cache:
                policy_probs, value = self.evaluation_cache[state_key]
                self.cache_hits += 1
            else:
                # 使用神经网络评估
                if hasattr(self.neural_network, 'predict_policy_value'):
                    policy_probs, value = self.neural_network.predict_policy_value(node.state, player)
                else:
                    # 向后兼容
                    policy_probs, value = self.neural_network.predict(node.state, player)
                
                # 缓存管理
                if len(self.evaluation_cache) > self.max_cache_size:
                    # 清理一半的缓存
                    items = list(self.evaluation_cache.items())
                    self.evaluation_cache = dict(items[len(items)//2:])
                
                self.evaluation_cache[state_key] = (policy_probs, value)
        
        # 为根节点添加Dirichlet噪声（用于探索）
        if self.add_dirichlet_noise and node.parent is None and policy_probs:
            policy_probs = self._add_dirichlet_noise(policy_probs)
        
        # 更新节点信息
        node.policy_probs = policy_probs
        node.value_estimate = value
    
    def _get_legal_actions(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """获取合法动作"""
        legal_actions = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:  # 空位
                    legal_actions.append((i, j))
        return legal_actions
    
    def _get_action_probabilities(self, root: MCTSNode) -> Dict[Tuple[int, int], float]:
        """
        计算基于访问次数的动作概率分布
        
        Args:
            root: 根节点
            
        Returns:
            动作概率分布
        """
        action_probs = {}
        total_visits = sum(child.visit_count for child in root.children.values())
        
        for action, child in root.children.items():
            if total_visits > 0:
                action_probs[action] = child.visit_count / total_visits
            else:
                action_probs[action] = 1.0 / len(root.children)
        
        return action_probs
    
    def _add_dirichlet_noise(self, policy_probs: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        为策略概率添加Dirichlet噪声（增强探索）
        
        Args:
            policy_probs: 原始策略概率
            
        Returns:
            添加噪声后的策略概率
        """
        if not policy_probs:
            return policy_probs
        
        actions = list(policy_probs.keys())
        probs = [policy_probs[action] for action in actions]
        
        # 生成Dirichlet噪声
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        # 混合原始概率和噪声（通常使用0.75的原始概率 + 0.25的噪声）
        epsilon = 0.25
        noisy_probs = {}
        for i, action in enumerate(actions):
            noisy_probs[action] = (1 - epsilon) * probs[i] + epsilon * noise[i]
        
        return noisy_probs
    
    def _get_state_key(self, state: np.ndarray, player: int) -> str:
        """生成状态缓存键"""
        return f"{state.tobytes()}_{player}"
    
    def clear_cache(self):
        """清理缓存"""
        self.evaluation_cache.clear()
        self.cache_hits = 0
    
    def get_statistics(self) -> Dict:
        """获取搜索统计信息"""
        return {
            'total_simulations': self.total_simulations,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.total_simulations),
            'cache_size': len(self.evaluation_cache),
            'c_puct': self.c_puct,
            'num_simulations': self.num_simulations
        }

class AlphaZeroPlayer:
    """AlphaZero风格的AI玩家"""
    
    def __init__(self, neural_network, mcts_simulations: int = 800, temperature: float = 1.0):
        """
        初始化AlphaZero玩家
        
        Args:
            neural_network: 神经网络模型
            mcts_simulations: MCTS模拟次数
            temperature: 温度参数（控制探索程度）
        """
        self.neural_network = neural_network
        self.mcts = NeuralMCTS(neural_network, num_simulations=mcts_simulations)
        self.temperature = temperature
        
        # 游戏历史
        self.move_history = []
        self.search_statistics = []
    
    def get_move(self, state: np.ndarray, player: int) -> Tuple[Tuple[int, int], Dict]:
        """
        获取最佳移动
        
        Args:
            state: 当前棋盘状态
            player: 当前玩家
            
        Returns:
            (move, search_info): 最佳移动和搜索信息
        """
        # 执行MCTS搜索
        action_probs, root_value = self.mcts.search(state, player)
        
        # 根据温度参数选择动作
        if self.temperature == 0:
            # 贪婪选择
            best_action = max(action_probs.keys(), key=lambda a: action_probs[a])
        else:
            # 根据概率分布采样
            actions = list(action_probs.keys())
            probs = [action_probs[a] ** (1 / self.temperature) for a in actions]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            
            best_action = np.random.choice(len(actions), p=probs)
            best_action = actions[best_action]
        
        # 记录搜索信息
        search_info = {
            'action_probabilities': action_probs,
            'root_value': root_value,
            'mcts_statistics': self.mcts.get_statistics(),
            'temperature': self.temperature
        }
        
        self.move_history.append((state.copy(), action_probs, best_action))
        self.search_statistics.append(search_info)
        
        return best_action, search_info
    
    def set_temperature(self, temperature: float):
        """设置温度参数"""
        self.temperature = temperature
    
    def reset_game(self):
        """重置游戏状态"""
        self.move_history = []
        self.search_statistics = []
    
    def create_root(self, board: List[List[int]], player: int) -> MCTSNode:
        """创建MCTS根节点"""
        state = np.array(board)
        root = MCTSNode(state=state, parent=None)
        return root
    
    def run_simulation(self, root: MCTSNode):
        """运行一次MCTS模拟"""
        # 选择
        leaf = self._select_node(root)
        
        # 扩展
        if not self._is_terminal(leaf.state):
            self._expand_node(leaf)
        
        # 模拟
        value = self._simulate(leaf)
        
        # 回传
        self._backpropagate(leaf, value)
    
    def get_move_probabilities(self, root: MCTSNode, temperature: float = 1.0) -> Dict[Tuple[int, int], float]:
        """获取移动概率分布"""
        if not root.children:
            return {}
        
        # 计算每个动作的访问概率
        total_visits = sum(child.visit_count for child in root.children.values())
        
        if total_visits == 0:
            return {}
        
        action_probs = {}
        for action, child in root.children.items():
            if temperature == 0:
                # 贪婪选择
                prob = 1.0 if child.visit_count == max(c.visit_count for c in root.children.values()) else 0.0
            else:
                # 根据访问次数和温度计算概率
                prob = (child.visit_count ** (1.0 / temperature)) / (total_visits ** (1.0 / temperature))
            
            action_probs[action] = prob
        
        # 归一化
        prob_sum = sum(action_probs.values())
        if prob_sum > 0:
            action_probs = {action: prob / prob_sum for action, prob in action_probs.items()}
        
        return action_probs
    
    def _select_node(self, node: MCTSNode) -> MCTSNode:
        """选择节点（PUCT算法）"""
        while node.is_expanded and not self._is_terminal(node.state):
            # 选择最佳子节点
            best_child = None
            best_value = -float('inf')
            
            for action, child in node.children.items():
                # PUCT公式
                prior_prob = node.policy_probs.get(action, 1.0 / len(node.children))
                exploration_value = self.c_puct * prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
                exploitation_value = child.average_value
                
                uct_value = exploitation_value + exploration_value
                
                if uct_value > best_value:
                    best_value = uct_value
                    best_child = child
            
            if best_child:
                node = best_child
            else:
                break
        
        return node
    
    def _expand_node(self, node: MCTSNode):
        """扩展节点"""
        # 获取合法动作
        legal_actions = self._get_legal_actions_from_state(node.state)
        
        for action in legal_actions:
            # 创建子节点
            new_state = self._apply_action(node.state, action)
            child = MCTSNode(state=new_state, parent=node)
            node.children[action] = child
    
    def _simulate(self, node: MCTSNode) -> float:
        """模拟游戏到结束"""
        if self._is_terminal(node.state):
            return self._get_game_result(node.state)
        
        # 使用网络价值估计作为模拟结果
        if hasattr(self.network, 'predict_policy_value'):
            _, value = self.network.predict_policy_value(node.state, player=1)
            return value
        
        # 简单的随机模拟
        return random.uniform(-1, 1)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """回传价值"""
        while node:
            node.visit_count += 1
            node.total_value += value
            value = -value  # 交替玩家的价值
            node = node.parent
    
    def _is_terminal(self, state: np.ndarray) -> bool:
        """检查是否为终端状态"""
        # 检查是否有五连
        for player in [1, 2]:
            if self._check_win(state, player):
                return True
        
        # 检查是否平局
        return np.sum(state == 0) == 0
    
    def _check_win(self, state: np.ndarray, player: int) -> bool:
        """检查指定玩家是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == player:
                    for dx, dy in directions:
                        if self._check_line_win(state, i, j, dx, dy, player):
                            return True
        
        return False
    
    def _check_line_win(self, state: np.ndarray, row: int, col: int, dx: int, dy: int, player: int) -> bool:
        """检查某个方向是否获胜"""
        count = 1
        
        # 正向检查
        x, y = row + dx, col + dy
        while (0 <= x < state.shape[0] and 0 <= y < state.shape[1] and state[x, y] == player):
            count += 1
            x += dx
            y += dy
        
        # 反向检查
        x, y = row - dx, col - dy
        while (0 <= x < state.shape[0] and 0 <= y < state.shape[1] and state[x, y] == player):
            count += 1
            x -= dx
            y -= dy
        
        return count >= 5
    
    def _get_game_result(self, state: np.ndarray) -> float:
        """获取游戏结果"""
        if self._check_win(state, 1):
            return 1.0
        elif self._check_win(state, 2):
            return -1.0
        else:
            return 0.0
    
    def _get_legal_actions_from_state(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """从状态获取合法动作"""
        legal_actions = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:
                    legal_actions.append((i, j))
        return legal_actions
    
    def _apply_action(self, state: np.ndarray, action: Tuple[int, int]) -> np.ndarray:
        """应用动作到状态"""
        new_state = state.copy()
        row, col = action
        # 假设当前玩家是1（实际使用时需要根据具体情况调整）
        new_state[row, col] = 1
        return new_state

# 神经网络接口基类
class NeuralNetworkInterface:
    """神经网络接口基类"""
    
    def predict_policy_value(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        预测策略概率和价值
        
        Args:
            state: 棋盘状态
            player: 当前玩家
            
        Returns:
            (policy_probs, value): 策略概率分布和价值估计
        """
        raise NotImplementedError
    
    def predict_policy(self, state: np.ndarray, player: int) -> Dict[Tuple[int, int], float]:
        """仅预测策略概率"""
        policy_probs, _ = self.predict_policy_value(state, player)
        return policy_probs
    
    def predict_value(self, state: np.ndarray, player: int) -> float:
        """仅预测价值"""
        _, value = self.predict_policy_value(state, player)
        return value

class AlphaZeroStyleNetwork(NeuralNetworkInterface):
    """AlphaZero风格的双头神经网络"""
    
    def __init__(self, policy_network=None, value_network=None, combined_network=None):
        """
        初始化网络
        
        Args:
            policy_network: 策略网络
            value_network: 价值网络  
            combined_network: 组合网络（包含策略+价值输出）
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.combined_network = combined_network
        
        if not any([policy_network, value_network, combined_network]):
            # 如果没有提供网络，使用dummy实现
            self.use_dummy = True
        else:
            self.use_dummy = False
    
    def predict_policy_value(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        预测策略概率和价值
        
        Args:
            state: 棋盘状态
            player: 当前玩家
            
        Returns:
            (policy_probs, value): 策略概率分布和价值估计
        """
        if self.use_dummy:
            return self._dummy_predict(state, player)
        
        if self.combined_network:
            # 使用组合网络
            return self._predict_combined(state, player)
        else:
            # 分别使用策略和价值网络
            policy_probs = self._predict_policy_separate(state, player)
            value = self._predict_value_separate(state, player)
            return policy_probs, value
    
    def _dummy_predict(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """dummy预测实现"""
        # 获取合法动作
        legal_actions = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:
                    legal_actions.append((i, j))
        
        # 均匀分布的策略概率
        prob = 1.0 / len(legal_actions) if legal_actions else 0.0
        policy_probs = {action: prob for action in legal_actions}
        
        # 随机价值估计
        value = random.uniform(-1, 1)
        
        return policy_probs, value
    
    def _predict_combined(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """使用组合网络预测"""
        # 这里应该调用实际的神经网络
        # 示例实现
        network_input = self._prepare_input(state, player)
        policy_output, value_output = self.combined_network.predict(network_input)
        
        # 转换为字典格式
        legal_actions = self._get_legal_actions(state)
        policy_probs = {}
        for i, action in enumerate(legal_actions):
            policy_probs[action] = policy_output[i] if i < len(policy_output) else 0.0
        
        return policy_probs, float(value_output)
    
    def _predict_policy_separate(self, state: np.ndarray, player: int) -> Dict[Tuple[int, int], float]:
        """使用独立策略网络预测"""
        if not self.policy_network:
            return self._dummy_predict(state, player)[0]
        
        network_input = self._prepare_input(state, player)
        policy_output = self.policy_network.predict(network_input)
        
        legal_actions = self._get_legal_actions(state)
        policy_probs = {}
        for i, action in enumerate(legal_actions):
            policy_probs[action] = policy_output[i] if i < len(policy_output) else 0.0
        
        return policy_probs
    
    def _predict_value_separate(self, state: np.ndarray, player: int) -> float:
        """使用独立价值网络预测"""
        if not self.value_network:
            return self._dummy_predict(state, player)[1]
        
        network_input = self._prepare_input(state, player)
        value_output = self.value_network.predict(network_input)
        
        return float(value_output)
    
    def _prepare_input(self, state: np.ndarray, player: int) -> np.ndarray:
        """准备网络输入"""
        # 这里应该实现特征提取
        # 示例：简单展平棋盘
        flattened = state.flatten()
        player_features = np.array([player])
        return np.concatenate([flattened, player_features])
    
    def _get_legal_actions(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """获取合法动作"""
        legal_actions = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:
                    legal_actions.append((i, j))
        return legal_actions
    
    def train_batch(self, board_states: np.ndarray, policy_targets: np.ndarray, 
                   value_targets: np.ndarray) -> Dict[str, float]:
        """
        训练批次数据
        
        Args:
            board_states: 棋盘状态批次
            policy_targets: 策略目标批次
            value_targets: 价值目标批次
            
        Returns:
            损失信息
        """
        if self.use_dummy:
            # Dummy网络返回虚拟损失
            return {
                'loss': random.uniform(0.1, 0.5),
                'policy_loss': random.uniform(0.05, 0.3),
                'value_loss': random.uniform(0.05, 0.2)
            }
        
        # 这里应该实现实际的训练逻辑
        # 由于当前使用dummy实现，返回虚拟损失
        return self._dummy_train_batch(board_states, policy_targets, value_targets)
    
    def _dummy_train_batch(self, board_states: np.ndarray, policy_targets: np.ndarray, 
                          value_targets: np.ndarray) -> Dict[str, float]:
        """虚拟训练实现"""
        # 模拟训练损失
        policy_loss = random.uniform(0.1, 0.4)
        value_loss = random.uniform(0.05, 0.2)
        total_loss = policy_loss + value_loss
        
        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
    
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测策略和价值（用于强化学习）
        
        Args:
            board_state: 棋盘状态
            
        Returns:
            (policy, value): 策略概率和价值估计
        """
        # 获取策略和价值
        policy_probs, value = self.predict_policy_value(board_state, player=1)
        
        # 转换策略为numpy数组
        board_size = board_state.shape[-1]
        policy_array = np.zeros((board_size * board_size,))
        
        for (row, col), prob in policy_probs.items():
            policy_array[row * board_size + col] = prob
        
        return policy_array, value
    
    def get_state(self) -> Dict[str, Any]:
        """获取网络状态（用于保存）"""
        return {
            'use_dummy': self.use_dummy,
            'dummy_weights': random.random() if self.use_dummy else None
        }
    
    def set_state(self, state: Dict[str, Any]):
        """设置网络状态（用于加载）"""
        self.use_dummy = state.get('use_dummy', True)
        # 这里应该加载实际的网络权重

# 向后兼容
class DummyNeuralNetwork(AlphaZeroStyleNetwork):
    """示例神经网络（用于演示接口）"""
    
    def __init__(self):
        super().__init__()
        self.use_dummy = True
    
    def predict(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """向后兼容的predict方法"""
        return self.predict_policy_value(state, player)

class NeuralEvaluatorNetworkAdapter(NeuralNetworkInterface):
    """
    将NeuralEvaluator适配为NeuralNetworkInterface接口
    """
    
    def __init__(self, neural_evaluator):
        self.neural_evaluator = neural_evaluator
    
    def predict_policy_value(self, state: np.ndarray, player: int) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        预测策略概率和价值
        
        Args:
            state: 棋盘状态
            player: 当前玩家
            
        Returns:
            (policy_probs, value): 策略概率分布和价值估计
        """
        # 转换为list格式
        board = state.tolist()
        
        # 使用神经网络评估器获取价值
        try:
            neural_score = self.neural_evaluator.evaluate_board(board, player)
            # 将分数转换为-1到1的范围
            value = max(-1.0, min(1.0, neural_score / 1000.0))
        except:
            value = 0.0
        
        # 生成策略概率（基于启发式方法）
        policy_probs = self._generate_policy_from_evaluator(board, player)
        
        return policy_probs, value
    
    def _generate_policy_from_evaluator(self, board: List[List[int]], player: int) -> Dict[Tuple[int, int], float]:
        """
        基于评估器生成策略概率
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            策略概率分布
        """
        # 获取合法动作
        legal_actions = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    legal_actions.append((i, j))
        
        if not legal_actions:
            return {}
        
        # 使用评估器对每个动作评分
        action_scores = []
        for action in legal_actions:
            row, col = action
            # 模拟落子
            board[row][col] = player
            
            try:
                score = self.neural_evaluator.evaluate_board(board, player)
                action_scores.append(score)
            except:
                action_scores.append(0)
            
            # 恢复棋盘
            board[row][col] = 0
        
        # 转换为概率分布（使用softmax）
        if not action_scores:
            # 均匀分布
            prob = 1.0 / len(legal_actions)
            return {action: prob for action in legal_actions}
        
        # Softmax转换
        max_score = max(action_scores)
        exp_scores = [math.exp((score - max_score) / 100.0) for score in action_scores]  # 温度=100
        sum_exp = sum(exp_scores)
        
        policy_probs = {}
        for i, action in enumerate(legal_actions):
            policy_probs[action] = exp_scores[i] / sum_exp if sum_exp > 0 else 1.0 / len(legal_actions)
        
        return policy_probs

class NeuralMCTSAdapter:
    """
    Neural MCTS适配器，兼容现有AI系统接口
    """
    
    def __init__(self, neural_network, mcts_simulations: int = 800, 
                 c_puct: float = 1.0, time_limit: float = 3.0):
        """
        初始化Neural MCTS适配器
        
        Args:
            neural_network: 神经网络模型
            mcts_simulations: MCTS模拟次数
            c_puct: PUCT探索常数
            time_limit: 时间限制
        """
        self.neural_network = neural_network
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.time_limit = time_limit
        
        # 初始化MCTS引擎
        self.mcts_engine = NeuralMCTS(
            neural_network=neural_network,
            c_puct=c_puct,
            num_simulations=mcts_simulations
        )
        
        # 初始化AlphaZero玩家
        self.alphazero_player = AlphaZeroPlayer(
            neural_network=neural_network,
            mcts_simulations=mcts_simulations,
            temperature=0.0  # 使用贪婪选择
        )
        
        # 统计信息
        self.total_moves = 0
        self.total_time = 0
        self.best_moves_history = []
    
    def get_best_move(self, board: List[List[int]], player: int):
        """
        获取最佳移动（兼容ModernGomokuAI接口）
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            SearchResult: 搜索结果
        """
        import time
        start_time = time.time()
        
        # 转换棋盘格式
        state = np.array(board)
        
        # 使用AlphaZero玩家获取移动
        try:
            move, search_info = self.alphazero_player.get_move(state, player)
            
            # 计算时间
            elapsed_time = time.time() - start_time
            
            # 更新统计
            self.total_moves += 1
            self.total_time += elapsed_time
            
            # 记录最佳移动
            self.best_moves_history.append({
                'move': move,
                'time': elapsed_time,
                'value': search_info.get('root_value', 0.0),
                'simulations': self.mcts_simulations
            })
            
            # 创建SearchResult类型的简单对象
            class SearchResult:
                def __init__(self, move, score, depth, time_elapsed, nodes_searched, 
                           best_path=None, pruned_branches=0, evaluation_cache_hits=0):
                    self.move = move
                    self.score = score
                    self.depth = depth
                    self.time_elapsed = time_elapsed
                    self.nodes_searched = nodes_searched
                    self.best_path = best_path or []
                    self.pruned_branches = pruned_branches
                    self.evaluation_cache_hits = evaluation_cache_hits
            
            result = SearchResult(
                move=move,
                score=int(search_info.get('root_value', 0.0) * 1000),  # 转换为整数分数
                depth=0,  # MCTS不使用深度概念
                time_elapsed=elapsed_time,
                nodes_searched=self.mcts_simulations,
                best_path=[move] if move else [],
                pruned_branches=0,
                evaluation_cache_hits=search_info.get('mcts_statistics', {}).get('cache_hits', 0)
            )
            
            return result
            
        except Exception as e:
            print(f"Neural MCTS错误: {e}")
            # 返回空结果
            class SearchResult:
                def __init__(self, move, score, depth, time_elapsed, nodes_searched, 
                           best_path=None, pruned_branches=0, evaluation_cache_hits=0):
                    self.move = move
                    self.score = score
                    self.depth = depth
                    self.time_elapsed = time_elapsed
                    self.nodes_searched = nodes_searched
                    self.best_path = best_path or []
                    self.pruned_branches = pruned_branches
                    self.evaluation_cache_hits = evaluation_cache_hits
            
            return SearchResult(
                move=None,
                score=0,
                depth=0,
                time_elapsed=time.time() - start_time,
                nodes_searched=0,
                best_path=[],
                pruned_branches=0,
                evaluation_cache_hits=0
            )
    
    def evaluate_board(self, board: List[List[int]], player: int) -> int:
        """
        评估棋盘状态（兼容评估器接口）
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            
        Returns:
            评估分数
        """
        state = np.array(board)
        
        try:
            # 使用神经网络获取价值估计
            if hasattr(self.neural_network, 'predict_value'):
                value = self.neural_network.predict_value(state, player)
            else:
                _, value = self.neural_network.predict(state, player)
            
            # 转换为整数分数（-1000到1000）
            return int(value * 1000)
            
        except Exception as e:
            print(f"Neural评估错误: {e}")
            return 0
    
    def set_data_collection(self, enabled: bool):
        """设置数据收集模式"""
        self.mcts.data_collection_enabled = enabled
    
    def get_search_info(self) -> Dict:
        """获取搜索信息用于训练数据收集"""
        if not self.mcts.data_collection_enabled:
            return {}
        
        return {
            'move_probabilities': self.mcts.move_probabilities,
            'visit_counts': self.mcts.visit_counts,
            'value_estimate': self.mcts.value_estimate
        }
    
    def get_move_suggestions(self, board: List[List[int]], player: int, top_k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """
        获取移动建议（按概率排序）
        
        Args:
            board: 棋盘状态
            player: 当前玩家
            top_k: 返回前几个建议
            
        Returns:
            移动建议列表 [(move, probability), ...]
        """
        state = np.array(board)
        
        try:
            # 执行MCTS搜索
            action_probs, _ = self.mcts_engine.search(state, player)
            
            # 按概率排序
            sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
            
            # 返回前 top_k 个
            return sorted_actions[:top_k]
            
        except Exception as e:
            print(f"获取移动建议错误: {e}")
            return []
    
    def set_temperature(self, temperature: float):
        """设置温度参数"""
        self.alphazero_player.set_temperature(temperature)
    
    def set_simulations(self, simulations: int):
        """设置模拟次数"""
        self.mcts_simulations = simulations
        self.mcts_engine.num_simulations = simulations
        
        # 重新初始化AlphaZero玩家
        self.alphazero_player = AlphaZeroPlayer(
            neural_network=self.neural_network,
            mcts_simulations=simulations,
            temperature=self.alphazero_player.temperature
        )
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        mcts_stats = self.mcts_engine.get_statistics()
        
        avg_time = self.total_time / max(1, self.total_moves)
        
        return {
            'adapter_type': 'Neural MCTS',
            'total_moves': self.total_moves,
            'total_time': self.total_time,
            'avg_time_per_move': avg_time,
            'simulations_per_move': self.mcts_simulations,
            'c_puct': self.c_puct,
            'mcts_statistics': mcts_stats,
            'move_history_count': len(self.best_moves_history)
        }
    
    def reset(self):
        """重置统计信息"""
        self.total_moves = 0
        self.total_time = 0
        self.best_moves_history = []
        self.mcts_engine.clear_cache()
        self.alphazero_player.reset_game()

# 测试代码
if __name__ == "__main__":
    # 创建测试环境
    dummy_network = DummyNeuralNetwork()
    player = AlphaZeroPlayer(dummy_network, mcts_simulations=100)
    
    # 创建测试棋盘
    board = np.zeros((15, 15), dtype=int)
    board[7, 7] = 1  # 黑棋
    board[7, 8] = 2  # 白棋
    
    # 测试移动
    move, info = player.get_move(board, 1)
    print(f"推荐移动: {move}")
    print(f"根节点价值: {info['root_value']:.3f}")
    print(f"MCTS统计: {info['mcts_statistics']}")

