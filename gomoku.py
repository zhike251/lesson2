import pygame
import sys
import random
import time
from integrated_ai import EnhancedGomokuGame, draw_ai_info, draw_ai_thinking, draw_last_ai_move

# 初始化pygame
pygame.init()

# 游戏常量
BOARD_SIZE = 15  # 15x15的棋盘
CELL_SIZE = 40   # 每个格子的大小
MARGIN = 60      # 边距
PIECE_RADIUS = 18  # 棋子半径

# 窗口大小
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN + 100

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (205, 170, 125)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)

# 新增美化的颜色
WOOD_BACKGROUND = (222, 184, 135)  # 浅木纹色
WOOD_BORDER = (160, 120, 80)     # 深木纹边框
STAR_COLOR = (255, 215, 0)        # 金色星位
PIECE_SHADOW = (50, 50, 50)      # 棋子阴影
HIGHLIGHT_COLOR = (255, 255, 0)  # 高亮颜色
INFO_PANEL_BG = (240, 240, 240)   # 信息面板背景
BUTTON_HOVER = (100, 150, 255)    # 按钮悬停色

# 创建窗口
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("五子棋 - 深度学习AI系统")

# 字体 - 使用系统字体支持中文
try:
    # 尝试使用系统中文字体
    title_font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 48)
    font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 36)
    small_font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 24)
    tiny_font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 20)
except:
    try:
        # 尝试使用微软雅黑
        title_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttf", 48)
        font = pygame.font.Font("C:/Windows/Fonts/msyh.ttf", 36)
        small_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttf", 24)
        tiny_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttf", 20)
    except:
        try:
            # 尝试使用宋体
            title_font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 48)
            font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 36)
            small_font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 24)
            tiny_font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 20)
        except:
            # 如果都无法加载，使用默认字体
            title_font = pygame.font.Font(None, 48)
            font = pygame.font.Font(None, 36)
            small_font = pygame.font.Font(None, 24)
            tiny_font = pygame.font.Font(None, 20)

# AI难度选项
DIFFICULTY_OPTIONS = [
    {"name": "简单", "key": "easy", "description": "适合初学者", "color": GREEN},
    {"name": "中等", "key": "medium", "description": "标准挑战", "color": BLUE},
    {"name": "困难", "key": "hard", "description": "高手级别", "color": RED},
    {"name": "专家", "key": "expert", "description": "极限挑战", "color": DARK_GRAY},
    {"name": "神经网络", "key": "neural", "description": "AI增强", "color": (128, 0, 128)},
    {"name": "强化学习", "key": "reinforced", "description": "自博弈训练", "color": (0, 128, 0)},
    {"name": "Neural MCTS", "key": "neural_mcts", "description": "顶级AI", "color": (255, 165, 0)},
    {"name": "终极威胁AI", "key": "ultimate_threat", "description": "最强AI 100%防守", "color": (255, 0, 255)}
]

class DifficultySelector:
    """难度选择界面"""
    
    def __init__(self, screen):
        self.screen = screen
        self.selected_difficulty = None
        self.buttons = []
        self.create_buttons()
        
    def create_buttons(self):
        """创建难度选择按钮"""
        button_width = 300
        button_height = 60
        button_spacing = 20
        start_y = 200
        
        for i, option in enumerate(DIFFICULTY_OPTIONS):
            x = (WINDOW_WIDTH - button_width) // 2
            y = start_y + i * (button_height + button_spacing)
            
            button_rect = pygame.Rect(x, y, button_width, button_height)
            self.buttons.append({
                'rect': button_rect,
                'option': option,
                'hover': False
            })
    
    def draw(self):
        """绘制难度选择界面"""
        # 绘制背景
        self.screen.fill(LIGHT_GRAY)
        
        # 绘制标题
        title_text = title_font.render("选择AI难度", True, BLACK)
        title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(title_text, title_rect)
        
        # 绘制副标题
        subtitle_text = small_font.render("深度学习五子棋AI系统", True, DARK_GRAY)
        subtitle_rect = subtitle_text.get_rect(center=(WINDOW_WIDTH // 2, 130))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # 绘制难度按钮
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.buttons:
            option = button['option']
            rect = button['rect']
            
            # 检查鼠标悬停
            button['hover'] = rect.collidepoint(mouse_pos)
            
            # 绘制按钮背景
            if button['hover']:
                pygame.draw.rect(self.screen, option['color'], rect)
                text_color = WHITE
            else:
                pygame.draw.rect(self.screen, option['color'], rect, 3)
                text_color = BLACK
            
            # 绘制按钮文字
            name_text = font.render(option['name'], True, text_color)
            name_rect = name_text.get_rect(center=(rect.centerx, rect.centery - 10))
            self.screen.blit(name_text, name_rect)
            
            # 绘制描述文字
            desc_text = tiny_font.render(option['description'], True, text_color)
            desc_rect = desc_text.get_rect(center=(rect.centerx, rect.centery + 15))
            self.screen.blit(desc_text, desc_rect)
        
        # 绘制底部提示
        hint_text = small_font.render("点击选择难度开始游戏", True, DARK_GRAY)
        hint_rect = hint_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50))
        self.screen.blit(hint_text, hint_rect)
    
    def handle_click(self, pos):
        """处理鼠标点击"""
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                self.selected_difficulty = button['option']['key']
                return self.selected_difficulty  # 返回选择的难度，而不是True
        return None
    
    def handle_hover(self, pos):
        """处理鼠标悬停"""
        for button in self.buttons:
            button['hover'] = button['rect'].collidepoint(pos)

class GomokuGame:
    def __init__(self, ai_difficulty="medium"):
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 1  # 1为人类玩家（黑棋），2为电脑玩家（白棋）
        self.game_over = False
        self.winner = 0
        self.human_turn = True
        
        # 当前难度信息
        self.ai_difficulty = ai_difficulty
        self.difficulty_info = self._get_difficulty_info()
        
        # 根据难度选择不同的AI系统
        if ai_difficulty == "ultimate_threat":
            from ultimate_threat_ai import UltimateThreatAI
            self.ai_system = UltimateThreatAI()
        else:
            from integrated_ai import IntegratedGomokuAI
            self.ai_system = IntegratedGomokuAI(ai_difficulty=ai_difficulty)
        
        # AI思考时间
        self.ai_thinking_time = 0
        
        # 游戏统计
        self.move_count = 0
        self.game_time = 0
        self.start_time = time.time()
        
    def _get_difficulty_info(self):
        """获取当前难度信息"""
        for option in DIFFICULTY_OPTIONS:
            if option['key'] == self.ai_difficulty:
                return option
        return DIFFICULTY_OPTIONS[1]  # 默认中等难度
    
    def change_difficulty(self, new_difficulty):
        """更改AI难度"""
        self.ai_difficulty = new_difficulty
        self.difficulty_info = self._get_difficulty_info()
        
        # 重新创建AI系统
        from integrated_ai import IntegratedGomokuAI
        self.ai_system = IntegratedGomokuAI(ai_difficulty=new_difficulty)
        
        # 重置AI思考时间
        self.ai_thinking_time = 0
        
        # 调试信息：验证难度设置
        print(f"[调试] 游戏难度: {self.ai_difficulty}")
        print(f"[调试] AI系统难度: {self.ai_system.ai_difficulty}")
        print(f"[调试] 显示名称: {self.difficulty_info['name']}")
        print(f"[成功] AI难度已更改为: {self.difficulty_info['name']}")
        
    def reset_game(self):
        """重置游戏"""
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        self.human_turn = True
        
        # 重置AI系统 - 根据难度选择不同的AI
        if self.ai_difficulty == "ultimate_threat":
            from ultimate_threat_ai import UltimateThreatAI
            self.ai_system = UltimateThreatAI()
        else:
            from integrated_ai import IntegratedGomokuAI
            self.ai_system = IntegratedGomokuAI(ai_difficulty=self.ai_difficulty)
        
        self.ai_thinking_time = 0
        
        # 重置游戏统计
        self.move_count = 0
        self.game_time = 0
        self.start_time = time.time()
        
    def is_valid_move(self, row, col):
        """检查落子是否有效"""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col] == 0
        return False
        
    def make_move(self, row, col):
        """落子"""
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.move_count += 1
            
            # 更新游戏时间
            self.game_time = time.time() - self.start_time
            
            if self.check_winner(row, col):
                self.game_over = True
                self.winner = self.current_player
            else:
                self.current_player = 3 - self.current_player  # 切换玩家（1->2, 2->1）
                self.human_turn = not self.human_turn
            return True
        return False
        
    def check_winner(self, row, col):
        """检查是否获胜"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、对角线
        player = self.board[row][col]
        
        for dx, dy in directions:
            count = 1
            
            # 向正方向检查
            x, y = row + dx, col + dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy
                
            # 向负方向检查
            x, y = row - dx, col - dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
                
            if count >= 5:
                return True
                
        return False
        
    def get_empty_positions(self):
        """获取所有空位置"""
        empty_positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    empty_positions.append((i, j))
        return empty_positions
        
    def ai_move(self):
        """现代化AI逻辑：根据AI类型使用不同的接口"""
        import time
        start_time = time.time()
        
        # 根据AI类型使用不同的接口
        if self.ai_difficulty == "ultimate_threat":
            # UltimateThreatAI使用 -1 作为白棋
            ai_player = -1
            move = self.ai_system.get_move(self.board, ai_player)
        else:
            # 其他AI使用 2 作为白棋
            ai_player = 2
            move = self.ai_system.get_ai_move(self.board, ai_player)
        
        self.ai_thinking_time = time.time() - start_time
        
        if move:
            row, col = move
            if self.make_move(row, col):
                return row, col
        
        # 如果AI系统失败，使用备用策略
        return self._fallback_ai_move()
    
    def _fallback_ai_move(self):
        """备用AI策略"""
        # 首先检查是否有必胜位置
        winning_move = self.find_winning_move(2)
        if winning_move:
            row, col = winning_move
            self.make_move(row, col)
            return row, col
        
        # 检查是否需要阻止对手获胜
        blocking_move = self.find_winning_move(1)
        if blocking_move:
            row, col = blocking_move
            self.make_move(row, col)
            return row, col
        
        # 使用简单的评分策略
        best_score = float('-inf')
        best_move = None
        
        candidates = self.get_best_move()
        
        for i, j in candidates:
            if self.board[i][j] == 0:
                self.board[i][j] = 2
                score = self.evaluate_position(i, j, 2)
                self.board[i][j] = 0
                
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
        
        if best_move:
            row, col = best_move
            self.make_move(row, col)
            return row, col
        
        return None, None
    
    def find_winning_move(self, player):
        """寻找必胜或必防的位置"""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    # 模拟落子
                    self.board[i][j] = player
                    if self.check_winner(i, j):
                        self.board[i][j] = 0  # 恢复
                        return (i, j)
                    self.board[i][j] = 0  # 恢复
        return None
    
    def find_threat_move(self, player):
        """寻找需要阻止的活四威胁"""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    # 模拟对手落子
                    self.board[i][j] = player
                    if self.has_open_four(i, j, player):
                        self.board[i][j] = 0  # 恢复
                        return (i, j)
                    self.board[i][j] = 0  # 恢复
        return None
    
    def find_urgent_move(self, player):
        """寻找需要阻止的活三威胁"""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    # 模拟对手落子
                    self.board[i][j] = player
                    if self.has_open_three(i, j, player):
                        self.board[i][j] = 0  # 恢复
                        return (i, j)
                    self.board[i][j] = 0  # 恢复
        return None
    
    def find_attack_move(self, player):
        """寻找进攻位置（形成活四）"""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    # 模拟自己落子
                    self.board[i][j] = player
                    if self.has_open_four(i, j, player):
                        self.board[i][j] = 0  # 恢复
                        return (i, j)
                    self.board[i][j] = 0  # 恢复
        return None
    
    def has_open_four(self, row, col, player):
        """检查是否有活四"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            if self.check_open_four_in_direction(row, col, dx, dy, player):
                return True
        return False
    
    def has_open_three(self, row, col, player):
        """检查是否有活三"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            if self.check_open_three_in_direction(row, col, dx, dy, player):
                return True
        return False
    
    def check_open_four_in_direction(self, row, col, dx, dy, player):
        """检查某个方向是否有活四"""
        # 活四模式：[0,2,2,2,2,0] 或类似
        pattern = self.get_pattern_for_check(row, col, dx, dy, player)
        
        # 检查是否包含活四模式
        for i in range(len(pattern) - 5):
            sub_pattern = pattern[i:i+6]
            if self.is_open_four_pattern(sub_pattern, player):
                return True
        return False
    
    def check_open_three_in_direction(self, row, col, dx, dy, player):
        """检查某个方向是否有活三"""
        # 活三模式：[0,2,2,2,0] 或类似
        pattern = self.get_pattern_for_check(row, col, dx, dy, player)
        
        # 检查是否包含活三模式
        for i in range(len(pattern) - 4):
            sub_pattern = pattern[i:i+5]
            if self.is_open_three_pattern(sub_pattern, player):
                return True
        return False
    
    def get_pattern_for_check(self, row, col, dx, dy, player):
        """获取用于检查的模式"""
        pattern = []
        # 向两个方向各延伸4格
        for direction in [-1, 1]:
            x, y = row, col
            for _ in range(4):
                x += dx * direction
                y += dy * direction
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    pattern.append(self.board[x][y])
                else:
                    pattern.append(-1)  # 边界
            x, y = row, col
        
        # 插入当前位置
        pattern.insert(4, player)
        return pattern
    
    def is_open_four_pattern(self, pattern, player):
        """检查是否是活四模式"""
        # 简化的活四检测
        count = 0
        open_ends = 0
        for val in pattern:
            if val == player:
                count += 1
            elif val == 0:
                open_ends += 1
        
        return count == 4 and open_ends >= 2
    
    def is_open_three_pattern(self, pattern, player):
        """检查是否是活三模式"""
        # 简化的活三检测
        count = 0
        open_ends = 0
        for val in pattern:
            if val == player:
                count += 1
            elif val == 0:
                open_ends += 1
        
        return count == 3 and open_ends >= 2
    
    def evaluate_position(self, row, col, player):
        """评估某个位置的分数"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、对角线
        
        for dx, dy in directions:
            line_score = self.evaluate_line(row, col, dx, dy, player)
            score += line_score
            
        return score
    
    def evaluate_line(self, row, col, dx, dy, player):
        """评估某个方向上的分数"""
        # 获取这个方向上的完整模式
        pattern = self.get_pattern(row, col, dx, dy, player)
        return self.score_pattern(pattern)
    
    def get_pattern(self, row, col, dx, dy, player):
        """获取某个方向上的棋型模式"""
        pattern = []
        
        # 向正方向延伸5格
        x, y = row, col
        for _ in range(5):
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                pattern.append(self.board[x][y])
            else:
                pattern.append(-1)  # 边界
            x += dx
            y += dy
        
        # 向负方向延伸4格（不包括当前位置）
        x, y = row - dx, col - dy
        negative_pattern = []
        for _ in range(4):
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                negative_pattern.append(self.board[x][y])
            else:
                negative_pattern.append(-1)  # 边界
            x -= dx
            y -= dy
        
        # 合并模式：负方向 + 正方向
        return negative_pattern[::-1] + pattern
    
    def score_pattern(self, pattern):
        """根据棋型模式评分"""
        # 定义各种棋型的分数（使用元组而不是列表）
        scores = {
            # 五连
            (2, 2, 2, 2, 2): 100000,
            # 活四
            (0, 2, 2, 2, 2, 0): 10000,
            # 冲四
            (0, 2, 2, 2, 2): 5000,
            (2, 2, 2, 2, 0): 5000,
            (2, 0, 2, 2, 2): 3000,
            (2, 2, 0, 2, 2): 3000,
            # 活三
            (0, 2, 2, 2, 0): 1000,
            (0, 2, 2, 0, 2, 0): 800,
            (0, 2, 0, 2, 2, 0): 800,
            # 眠三
            (0, 2, 2, 2): 400,
            (2, 2, 2, 0): 400,
            (2, 2, 0, 2): 300,
            (2, 0, 2, 2): 300,
            # 活二
            (0, 2, 2, 0): 200,
            (0, 2, 0, 2, 0): 150,
            # 眠二
            (0, 2, 2): 50,
            (2, 2, 0): 50,
            (0, 2, 0, 2): 30,
            (2, 0, 2): 30,
        }
        
        max_score = 0
        # 检查所有可能的子模式
        for length in range(5, len(pattern) + 1):
            for i in range(len(pattern) - length + 1):
                sub_pattern = tuple(pattern[i:i+length])
                if sub_pattern in scores:
                    max_score = max(max_score, scores[sub_pattern])
        
        return max_score
    
    def get_best_move(self):
        """获取最佳落子位置（考虑周围有棋子的位置）"""
        candidates = []
        
        # 只考虑周围有棋子的空位
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0 and self.has_neighbor(i, j):
                    candidates.append((i, j))
        
        # 如果没有邻居，选择中心位置
        if not candidates:
            center = BOARD_SIZE // 2
            return [(center, center)]
        
        return candidates
    
    def has_neighbor(self, row, col):
        """检查某个位置周围是否有棋子"""
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                ni, nj = row + i, col + j
                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                    if self.board[ni][nj] != 0:
                        return True
        return False
    
    def get_ai_info(self):
        """获取AI信息"""
        # 检查AI系统是否有get_ai_info方法
        if hasattr(self.ai_system, 'get_ai_info'):
            return self.ai_system.get_ai_info()
        else:
            # 为UltimateThreatAI提供默认信息
            return {
                'name': 'UltimateThreatAI',
                'difficulty': 'ultimate_threat',
                'description': '最强AI 100%防守',
                'thinking_time': getattr(self, 'ai_thinking_time', 0),
                'threat_stats': getattr(self.ai_system, 'threat_stats', {
                    'threats_detected': 0,
                    'threats_defended': 0,
                    'defense_rate': 0.0
                })
            }
    
    def get_performance_stats(self):
        """获取性能统计"""
        ai_stats = self.ai_system.get_performance_summary()
        
        # 检查ai_stats是否已经包含嵌套的ai_stats字段
        if isinstance(ai_stats, dict) and 'ai_stats' in ai_stats:
            # 如果已经是嵌套结构，直接使用
            final_ai_stats = ai_stats['ai_stats']
        else:
            # 如果是扁平结构，包装成嵌套结构
            final_ai_stats = ai_stats
        
        return {
            'ai_stats': final_ai_stats,
            'game_stats': {
                'total_moves': len(getattr(self.ai_system, 'move_history', [])),
                'total_time': getattr(self.ai_system, 'performance_stats', {}).get('total_time', 0.0)
            }
        }

def draw_board(screen):
    """绘制棋盘"""
    # 绘制木纹背景
    screen.fill(WOOD_BACKGROUND)
    
    # 绘制棋盘边框
    board_rect = pygame.Rect(MARGIN - 15, MARGIN - 15, 
                            (BOARD_SIZE - 1) * CELL_SIZE + 30, 
                            (BOARD_SIZE - 1) * CELL_SIZE + 30)
    pygame.draw.rect(screen, WOOD_BORDER, board_rect)
    pygame.draw.rect(screen, BLACK, board_rect, 3)
    
    # 绘制网格线
    for i in range(BOARD_SIZE):
        # 横线
        start_x = MARGIN
        end_x = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE
        y = MARGIN + i * CELL_SIZE
        pygame.draw.line(screen, BLACK, (start_x, y), (end_x, y), 2)
        
        # 竖线
        x = MARGIN + i * CELL_SIZE
        start_y = MARGIN
        end_y = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE
        pygame.draw.line(screen, BLACK, (x, start_y), (x, end_y), 2)
    
    # 绘制星位（天元和星）
    star_positions = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
    for row, col in star_positions:
        center_x = MARGIN + col * CELL_SIZE
        center_y = MARGIN + row * CELL_SIZE
        pygame.draw.circle(screen, BLACK, (center_x, center_y), 6)
        pygame.draw.circle(screen, STAR_COLOR, (center_x, center_y), 4)
    
    # 绘制坐标标记
    draw_board_coordinates(screen)

def draw_board_coordinates(screen):
    """绘制棋盘坐标"""
    # 字体大小
    coord_font = tiny_font
    
    # 绘制字母坐标（A-O）
    letters = 'ABCDEFGHIJKLMNO'
    for i, letter in enumerate(letters):
        x = MARGIN + i * CELL_SIZE
        # 上方坐标
        text = coord_font.render(letter, True, BLACK)
        text_rect = text.get_rect(center=(x, MARGIN - 25))
        screen.blit(text, text_rect)
        # 下方坐标
        text_rect = text.get_rect(center=(x, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE + 25))
        screen.blit(text, text_rect)
    
    # 绘制数字坐标（1-15）
    for i in range(1, BOARD_SIZE + 1):
        y = MARGIN + (i - 1) * CELL_SIZE
        # 左侧坐标
        text = coord_font.render(str(i), True, BLACK)
        text_rect = text.get_rect(center=(MARGIN - 25, y))
        screen.blit(text, text_rect)
        # 右侧坐标
        text_rect = text.get_rect(center=(MARGIN + (BOARD_SIZE - 1) * CELL_SIZE + 25, y))
        screen.blit(text, text_rect)

def draw_pieces(screen, board):
    """绘制棋子"""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            center_x = MARGIN + j * CELL_SIZE
            center_y = MARGIN + i * CELL_SIZE
            
            if board[i][j] == 1:  # 黑棋
                # 绘制阴影
                shadow_offset = 2
                pygame.draw.circle(screen, PIECE_SHADOW, 
                                 (center_x + shadow_offset, center_y + shadow_offset), 
                                 PIECE_RADIUS)
                
                # 绘制黑棋主体
                pygame.draw.circle(screen, BLACK, (center_x, center_y), PIECE_RADIUS)
                
                # 添加高光效果
                highlight_x = center_x - PIECE_RADIUS // 3
                highlight_y = center_y - PIECE_RADIUS // 3
                pygame.draw.circle(screen, (40, 40, 40), 
                                 (highlight_x, highlight_y), PIECE_RADIUS // 4)
                
            elif board[i][j] == 2:  # 白棋
                # 绘制阴影
                shadow_offset = 2
                pygame.draw.circle(screen, PIECE_SHADOW, 
                                 (center_x + shadow_offset, center_y + shadow_offset), 
                                 PIECE_RADIUS)
                
                # 绘制白棋主体
                pygame.draw.circle(screen, WHITE, (center_x, center_y), PIECE_RADIUS)
                pygame.draw.circle(screen, BLACK, (center_x, center_y), PIECE_RADIUS, 2)
                
                # 添加高光效果
                highlight_x = center_x - PIECE_RADIUS // 3
                highlight_y = center_y - PIECE_RADIUS // 3
                pygame.draw.circle(screen, WHITE, 
                                 (highlight_x, highlight_y), PIECE_RADIUS // 3)

def draw_status(screen, game):
    """绘制游戏状态"""
    status_y = MARGIN + BOARD_SIZE * CELL_SIZE + 20
    
    # 绘制信息面板背景
    info_panel_height = 80
    info_panel_rect = pygame.Rect(5, 5, 200, info_panel_height)
    pygame.draw.rect(screen, (240, 240, 240), info_panel_rect)
    pygame.draw.rect(screen, BLACK, info_panel_rect, 2)
    
    # 绘制当前难度信息
    difficulty_text = tiny_font.render(f"AI难度: {game.difficulty_info['name']}", True, game.difficulty_info['color'])
    screen.blit(difficulty_text, (15, 15))
    
    # 调试信息：每帧都显示当前状态
    if hasattr(game, '_debug_counter'):
        game._debug_counter += 1
    else:
        game._debug_counter = 0
    
    # 每60帧显示一次调试信息（约1秒）
    if game._debug_counter % 60 == 0:
        print(f"[UI调试] 显示难度: {game.difficulty_info['name']}")
        print(f"[UI调试] 游戏难度: {game.ai_difficulty}")
        # 安全地获取AI系统难度信息
        if hasattr(game.ai_system, 'ai_difficulty'):
            print(f"[UI调试] AI系统难度: {game.ai_system.ai_difficulty}")
        else:
            print(f"[UI调试] AI系统类型: {type(game.ai_system).__name__}")
    
    # 绘制游戏统计信息
    if hasattr(game, 'move_count'):
        moves_text = tiny_font.render(f"步数: {game.move_count}", True, BLACK)
        screen.blit(moves_text, (15, 35))
    
    if hasattr(game, 'game_time') and game.game_time > 0:
        time_text = tiny_font.render(f"时间: {game.game_time:.1f}秒", True, BLACK)
        screen.blit(time_text, (15, 55))
    
    # 绘制游戏状态面板
    status_panel_width = 400
    status_panel_height = 100
    status_panel_rect = pygame.Rect(WINDOW_WIDTH // 2 - status_panel_width // 2, status_y - 10, 
                                   status_panel_width, status_panel_height)
    pygame.draw.rect(screen, (250, 250, 250), status_panel_rect)
    pygame.draw.rect(screen, BLACK, status_panel_rect, 2)
    
    # 绘制游戏状态
    if game.game_over:
        if game.winner == 1:
            text = "🎉 你赢了！"
            color = GREEN
        else:
            text = "💻 电脑赢了！"
            color = RED
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, status_y + 5))
    else:
        if game.human_turn:
            text = "🎯 你的回合（黑棋）"
            color = BLUE
        else:
            text = "🤖 电脑思考中...（白棋）"
            color = RED
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, status_y + 5))
        
        # 显示AI思考时间
        if not game.human_turn and game.ai_thinking_time > 0:
            time_text = small_font.render(f"思考时间: {game.ai_thinking_time:.2f}秒", True, RED)
            screen.blit(time_text, (WINDOW_WIDTH // 2 - time_text.get_width() // 2, status_y + 35))
    
    # 绘制AI信息
    draw_ai_info(screen, game, font, small_font)
    
    # 绘制按钮
    buttons = []
    
    # 重新开始按钮
    button_y = status_y + 60 if game.game_over else status_y + 50
    restart_rect = pygame.Rect(WINDOW_WIDTH // 2 - 80, button_y, 160, 40)
    
    # 检查鼠标悬停
    mouse_pos = pygame.mouse.get_pos()
    restart_hover = restart_rect.collidepoint(mouse_pos)
    
    # 绘制按钮背景
    if restart_hover:
        pygame.draw.rect(screen, (230, 230, 250), restart_rect)
        pygame.draw.rect(screen, BLUE, restart_rect, 3)
    else:
        pygame.draw.rect(screen, (240, 240, 240), restart_rect)
        pygame.draw.rect(screen, BLACK, restart_rect, 2)
    
    restart_text = small_font.render("重新开始", True, BLUE if restart_hover else BLACK)
    text_rect = restart_text.get_rect(center=restart_rect.center)
    screen.blit(restart_text, text_rect)
    buttons.append(('restart', restart_rect))
    
    # 难度选择按钮
    difficulty_rect = pygame.Rect(WINDOW_WIDTH - 120, 10, 100, 30)
    diff_hover = difficulty_rect.collidepoint(mouse_pos)
    
    # 绘制按钮背景
    if diff_hover:
        pygame.draw.rect(screen, (230, 240, 250), difficulty_rect)
        pygame.draw.rect(screen, GREEN, difficulty_rect, 2)
    else:
        pygame.draw.rect(screen, (240, 240, 240), difficulty_rect)
        pygame.draw.rect(screen, GRAY, difficulty_rect, 2)
    
    diff_text = tiny_font.render("选难度", True, GREEN if diff_hover else BLACK)
    text_rect = diff_text.get_rect(center=difficulty_rect.center)
    screen.blit(diff_text, text_rect)
    buttons.append(('difficulty', difficulty_rect))
    
    # 如果游戏进行中，显示提示信息
    if not game.game_over and game.human_turn:
        hint_text = tiny_font.render("💡 点击棋盘下棋", True, GRAY)
        screen.blit(hint_text, (WINDOW_WIDTH // 2 - hint_text.get_width() // 2, status_y + 75))
    
    return buttons

def get_board_position(mouse_x, mouse_y):
    """将鼠标位置转换为棋盘坐标"""
    col = round((mouse_x - MARGIN) / CELL_SIZE)
    row = round((mouse_y - MARGIN) / CELL_SIZE)
    return row, col

def main():
    clock = pygame.time.Clock()
    
    # 创建难度选择器
    difficulty_selector = DifficultySelector(screen)
    
    # 首先显示难度选择界面
    selected_difficulty = None
    selecting_difficulty = True
    
    while selecting_difficulty:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    selected_difficulty = difficulty_selector.handle_click(event.pos)
                    if selected_difficulty is not None:  # 检查是否为None而不是检查真值
                        selecting_difficulty = False
            
            elif event.type == pygame.MOUSEMOTION:
                difficulty_selector.handle_hover(event.pos)
        
        # 绘制难度选择界面
        screen.fill(WHITE)
        difficulty_selector.draw()
        
        pygame.display.flip()
        clock.tick(60)
    
    # 根据选择的难度创建游戏
    game = GomokuGame(ai_difficulty=selected_difficulty)
    
    running = True
    show_difficulty_selector = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    
                    # 绘制状态栏获取按钮
                    buttons = draw_status(screen, game)
                    
                    # 检查按钮点击
                    for button_type, rect in buttons:
                        if rect.collidepoint(mouse_x, mouse_y):
                            if button_type == 'restart':
                                game.reset_game()
                            elif button_type == 'difficulty':
                                show_difficulty_selector = True
                                break
                    
                    if show_difficulty_selector:
                        # 显示难度选择界面
                        new_difficulty = None
                        selecting = True
                        
                        while selecting:
                            for select_event in pygame.event.get():
                                if select_event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()
                                    
                                elif select_event.type == pygame.MOUSEBUTTONDOWN:
                                    if select_event.button == 1:
                                        new_difficulty = difficulty_selector.handle_click(select_event.pos)
                                        if new_difficulty is not None:  # 检查是否为None而不是检查真值
                                            selecting = False
                                            show_difficulty_selector = False
                                            game.change_difficulty(new_difficulty)
                                
                                elif select_event.type == pygame.MOUSEMOTION:
                                    difficulty_selector.handle_hover(select_event.pos)
                            
                            # 绘制难度选择界面
                            screen.fill(WHITE)
                            difficulty_selector.draw()
                            
                            pygame.display.flip()
                            clock.tick(60)
                    
                    elif game.human_turn and not game.game_over:
                        # 人类玩家落子
                        row, col = get_board_position(mouse_x, mouse_y)
                        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                            game.make_move(row, col)
        
        # 电脑玩家落子
        if not game.human_turn and not game.game_over:
            pygame.time.wait(500)  # 稍微延迟，让玩家看清
            game.ai_move()
        
        # 更新游戏时间
        if not game.game_over:
            game.game_time = time.time() - game.start_time
        
        # 绘制界面
        draw_board(screen)
        draw_pieces(screen, game.board)
        draw_status(screen, game)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()