import pygame, sys

pygame.init()
W, H = 300, 360
LW = 5
R, C = 3, 3
SZ = W // C
CR = SZ // 3
CW = 15
XW = 20
SP = SZ // 4

BG = (28, 170, 156) #random colors idk
LN = (123, 145, 115)
OC = (219, 31, 220)
XC = (166, 96, 46)
BTN_COLOR = (200, 200, 200)
BTN_TEXT = (50, 50, 50)

scr = pygame.display.set_mode((W, H))
pygame.display.set_caption('Tic Tac Toe')

font = pygame.font.SysFont(None, 30)

def draw_lines():
    scr.fill(BG)
    for r in range(1, R):
        pygame.draw.line(scr, LN, (0, r * SZ), (W, r * SZ), LW)
    for c in range(1, C):
        pygame.draw.line(scr, LN, (c * SZ, 0), (c * SZ, SZ * R), LW)

def draw_figs():
    for r in range(R):
        for c in range(C):
            if brd[r][c] == 'O':
                pygame.draw.circle(scr, OC, (c * SZ + SZ//2, r * SZ + SZ//2), CR, CW)
            elif brd[r][c] == 'X':
                pygame.draw.line(scr, XC, (c * SZ + SP, r * SZ + SZ - SP), (c * SZ + SZ - SP, r * SZ + SP), XW)
                pygame.draw.line(scr, XC, (c * SZ + SP, r * SZ + SP), (c * SZ + SZ - SP, r * SZ + SZ - SP), XW)

def draw_button():
    pygame.draw.rect(scr, BTN_COLOR, (100, 310, 100, 40))
    txt = font.render("Reset", True, BTN_TEXT)
    scr.blit(txt, (125, 320))

def win(p):
    for row in brd:
        if all(cell == p for cell in row): return True
    for c in range(C):
        if all(brd[r][c] == p for r in range(R)): return True
    if all(brd[i][i] == p for i in range(R)): return True
    if all(brd[i][R - i - 1] == p for i in range(R)): return True
    return False

def is_draw():
    return all(brd[r][c] != '' for r in range(R) for c in range(C))

def moves():
    return [(r, c) for r in range(R) for c in range(C) if brd[r][c] == '']

def ai():
    for r, c in moves():
        brd[r][c] = 'O'
        if win('O'): return
        brd[r][c] = ''
    for r, c in moves():
        brd[r][c] = 'X'
        if win('X'):
            brd[r][c] = 'O'
            return
        brd[r][c] = ''
    if brd[1][1] == '':
        brd[1][1] = 'O'
        return
    for r, c in [(0,0), (0,2), (2,0), (2,2)]:
        if brd[r][c] == '':
            brd[r][c] = 'O'
            return
    for r, c in [(0,1), (1,0), (1,2), (2,1)]:
        if brd[r][c] == '':
            brd[r][c] = 'O'
            return

def reset_game():
    global brd, pt, over
    brd = [['' for _ in range(C)] for _ in range(R)]
    pt = True
    over = False
    draw_lines()
    draw_button()

brd = [['' for _ in range(C)] for _ in range(R)]
pt = True
over = False
draw_lines()
draw_button()

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if e.type == pygame.MOUSEBUTTONDOWN:
            x, y = e.pos
            if 100 <= x <= 200 and 310 <= y <= 350:
                reset_game()
            elif y < SZ * R and not over and pt:
                r, c = y // SZ, x // SZ
                if brd[r][c] == '':
                    brd[r][c] = 'X'
                    pt = False
                    if win('X'):
                        print("Anay wins!")
                        over = True
                    elif is_draw():
                        print("draw!")
                        over = True
    if not pt and not over:
        ai()
        pt = True
        if win('O'):
            print("Opponent wins!")
            over = True
        elif is_draw():
            print("draw!")
            over = True
    draw_figs()
    draw_button()
    pygame.display.update()
