import pygame
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from pathfinding.core.grid import Grid
import warnings
warnings.filterwarnings('ignore')



WIDTH = 840
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("MY FIFRST TRY")
pygame.mixer.init()

BG = (235, 240, 242)
#BARRIER = (217, 163, 132)
BARRIER = (0,0,0)
GREY = (128, 128, 128)


class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row*width
        self.y = col*width
        self.color = BG
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.color == BARRIER

    def reset(self):
        self.color = BG



    def make_barrier(self):
        self.color = BARRIER








    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))




def make_grid(rows, width):
    grid=[]
    gap = width// rows
    for i in range(rows):
        grid.append([])
        for j in range (rows):
            node = Node(i,j, gap, rows)
            grid[i].append(node)

    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw (win, grid, rows, width):
    win.fill(BG)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width// rows
    y, x = pos

    row = y//gap
    col = x//gap

    return row, col
def guess(drawn):
    model = tf.keras.models.load_model('model1.model')
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_test[0] = tf.constant(drawn)
    predictions = np.argmax(model.predict(x_test[:1]), axis=1)

    print("I predict this number is a:", predictions)

    #plt.imshow(li[0], cmap=plt.cm.binary)
    #plt.show()

def main(win, width):
    ROWS = 28
    grid = make_grid(ROWS, width)

    run = True
    started = False


    arr = [[0 for i in range(ROWS)] for j in range(ROWS)]



    while run:
        draw(win, grid, ROWS, width)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]: #left button

                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.make_barrier()
                arr[col][row] = 255

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                  #print(arr)
                  guess(arr)

                if event.key == pygame.K_c:


                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                    arr = [[0 for i in range(ROWS)] for j in range(ROWS)]


    pygame.quit()

pygame.init()
main(WIN, WIDTH)
