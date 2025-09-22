import pygame
import sys
import random

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

# 创建窗口
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("五子棋")

# 字体 - 使用系统字体支持中文
try:
    # 尝试使用系统中文字体
    font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 36)
    small_font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 24)
except:
    try:
        # 尝试使用微软雅黑
        font = pygame.font.Font("C:/Windows/Fonts/msyh.ttf", 36)
        small_font = pygame.font.Font("C:/Windows/Fonts/msyh.ttf", 24)
    except:
        try:
            # 尝试使用宋体
            font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 36)
            small_font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 24)
        except:
            # 如果都无法加载，使用默认字体
            font = pygame.font.Font(None, 36)
            small_font = pygame.font.Font(None, 24)

class GomokuGame:
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 1  # 1为人类玩家（黑棋），2为电脑玩家（白棋）
        self.game_over = False
        self.winner = 0
        self.human_turn = True
        
    def reset_game(self):
        """重置游戏"""
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        self.human_turn = True
        
    def is_valid_move(self, row, col):
        """检查落子是否有效"""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row][col] == 0
        return False
        
    def make_move(self, row, col):
        """落子"""
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
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
        """智能AI逻辑：使用评分策略"""
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
        
        # 检查是否需要阻止对手形成活四
        threat_move = self.find_threat_move(1)
        if threat_move:
            row, col = threat_move
            self.make_move(row, col)
            return row, col
        
        # 检查是否需要阻止对手形成活三
        urgent_move = self.find_urgent_move(1)
        if urgent_move:
            row, col = urgent_move
            self.make_move(row, col)
            return row, col
        
        # 检查自己是否能形成活四
        attack_move = self.find_attack_move(2)
        if attack_move:
            row, col = attack_move
            self.make_move(row, col)
            return row, col
        
        # 使用评分策略选择最佳位置
        best_score = float('-inf')
        best_move = None
        
        # 获取候选位置（只考虑周围有棋子的位置）
        candidates = self.get_best_move()
        
        for i, j in candidates:
            if self.board[i][j] == 0:
                # 模拟落子
                self.board[i][j] = 2
                score = self.evaluate_position(i, j, 2)
                self.board[i][j] = 0  # 恢复
                
                # 检查这个位置对玩家的重要性
                self.board[i][j] = 1
                opponent_score = self.evaluate_position(i, j, 1)
                self.board[i][j] = 0  # 恢复
                
                # 综合评分：进攻 + 防守
                total_score = score + opponent_score * 1.5  # 提高防守权重
                
                # 添加位置权重（中心位置更有价值）
                center_bonus = (7 - abs(i - 7)) + (7 - abs(j - 7))
                total_score += center_bonus * 5
                
                if total_score > best_score:
                    best_score = total_score
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

def draw_board(screen):
    """绘制棋盘"""
    screen.fill(BROWN)
    
    # 绘制网格线
    for i in range(BOARD_SIZE):
        # 横线
        pygame.draw.line(screen, BLACK, 
                        (MARGIN, MARGIN + i * CELL_SIZE), 
                        (MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, MARGIN + i * CELL_SIZE), 2)
        # 竖线
        pygame.draw.line(screen, BLACK, 
                        (MARGIN + i * CELL_SIZE, MARGIN), 
                        (MARGIN + i * CELL_SIZE, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE), 2)
    
    # 绘制星位
    star_positions = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
    for row, col in star_positions:
        pygame.draw.circle(screen, BLACK, 
                          (MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE), 5)

def draw_pieces(screen, board):
    """绘制棋子"""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:  # 黑棋
                pygame.draw.circle(screen, BLACK, 
                                  (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE), 
                                  PIECE_RADIUS)
            elif board[i][j] == 2:  # 白棋
                pygame.draw.circle(screen, WHITE, 
                                  (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE), 
                                  PIECE_RADIUS)
                pygame.draw.circle(screen, BLACK, 
                                  (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE), 
                                  PIECE_RADIUS, 2)

def draw_status(screen, game):
    """绘制游戏状态"""
    status_y = MARGIN + BOARD_SIZE * CELL_SIZE + 20
    
    if game.game_over:
        if game.winner == 1:
            text = "你赢了！"
            color = BLACK
        else:
            text = "电脑赢了！"
            color = RED
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, status_y))
    else:
        if game.human_turn:
            text = "你的回合（黑棋）"
            color = BLACK
        else:
            text = "电脑思考中...（白棋）"
            color = RED
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, status_y))
    
    # 绘制重新开始按钮
    button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 80, status_y + 40, 160, 40)
    pygame.draw.rect(screen, BLACK, button_rect, 2)
    button_text = small_font.render("重新开始", True, BLACK)
    screen.blit(button_text, (button_rect.x + 40, button_rect.y + 10))
    
    return button_rect

def get_board_position(mouse_x, mouse_y):
    """将鼠标位置转换为棋盘坐标"""
    col = round((mouse_x - MARGIN) / CELL_SIZE)
    row = round((mouse_y - MARGIN) / CELL_SIZE)
    return row, col

def main():
    clock = pygame.time.Clock()
    game = GomokuGame()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    
                    # 检查是否点击重新开始按钮
                    status_y = MARGIN + BOARD_SIZE * CELL_SIZE + 20
                    button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 80, status_y + 40, 160, 40)
                    
                    if button_rect.collidepoint(mouse_x, mouse_y):
                        game.reset_game()
                    elif game.human_turn and not game.game_over:
                        # 人类玩家落子
                        row, col = get_board_position(mouse_x, mouse_y)
                        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                            game.make_move(row, col)
        
        # 电脑玩家落子
        if not game.human_turn and not game.game_over:
            pygame.time.wait(500)  # 稍微延迟，让玩家看清
            game.ai_move()
        
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