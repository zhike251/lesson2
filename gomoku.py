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
        """简单的AI逻辑：随机选择一个空位置"""
        empty_positions = self.get_empty_positions()
        if empty_positions:
            row, col = random.choice(empty_positions)
            self.make_move(row, col)
            return row, col
        return None, None

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