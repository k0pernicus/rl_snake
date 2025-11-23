import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()

font = pygame.font.Font(None, 25)
if font == None:
    print("Failed to load font!")
    exit(-1)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rewards
GOOD_REWARD = 10
BAD_REWARD  = -10
NO_REWARD   = 0

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN_FILL = (80, 255, 80)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20

STATS_DASHBOARD_WIDTH = 320

class SnakeGame:

    def __init__(self, game_width=1280, game_height=720):
        self._game_w = game_width
        self._game_h = game_height
        self.w = game_width + STATS_DASHBOARD_WIDTH
        self.h = game_height
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('RL Snake')
        self.clock = pygame.time.Clock()
        self._n_steps = 0
        self._n_games = 1

        self._plot_scores = []
        self._plot_mean_scores = []

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self._game_w/2, self._game_h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self._n_steps = 0

    def _place_food(self):
        x = random.randint(0, (self._game_w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self._game_h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def update_plots(self, n_games, scores, mean_scores):
        self._n_games = n_games
        self._plot_scores = scores
        self._plot_mean_scores = mean_scores

    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

        # Keep the distance between the snake and the food
        old_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        self._move(action) # update the head
        self.snake.insert(0, self.head)

        game_over = False
        # TODO : check with the length of the snake ( it has to eat... )
        if self._is_collision() or self._n_steps > 30*len(self.snake):
            game_over = True
            return BAD_REWARD, game_over, self.score

        reward = NO_REWARD
        if self.head == self.food:
            reward = GOOD_REWARD
            self.score += 1
            self._place_food()
        else:
            # Calculate new distance
            new_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            self.snake.pop()

            # The difference is to avoid the "vibrating snake" behavior : zero-sum getting closer, far, then close, then far, ...
            if new_dist < old_dist:
                reward = 1 # Reward only if the snake is now closer to the food
            else:
                reward = -2 # Do not reward if the snake is away from the food

        self._update_ui()
        self.clock.tick(SPEED)

        self._n_steps += 1

        return reward, game_over, self.score

    def _is_collision(self, coordinate=None):
        if coordinate is None: coordinate = self.head
        # hits boundary
        return (coordinate.x > self._game_w - BLOCK_SIZE or coordinate.x < 0 or coordinate.y > self._game_h - BLOCK_SIZE or coordinate.y < 0) or (coordinate in self.snake[1:])

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN_FILL, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        # pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.circle(surface=self.display, color=RED, center=(self.food.x + (BLOCK_SIZE // 2), self.food.y + (BLOCK_SIZE // 2)), radius=BLOCK_SIZE // 2, width=BLOCK_SIZE)
        pygame.draw.line(self.display, WHITE, (self._game_w, 0), (self._game_w, self.h), 2)

        self._draw_charts()

        pygame.display.flip()

    def _draw_charts(self):
        chart_x = self._game_w + 20
        chart_y = 50
        chart_w = 280
        chart_h = 200

        pygame.draw.rect(self.display, (20, 20, 20), (chart_x, chart_y, chart_w, chart_h))
        pygame.draw.rect(self.display, WHITE, (chart_x, chart_y, chart_w, chart_h), 1)

        if len(self._plot_scores) == 0:
            text_run_nb = font.render(f"Nb run: {self._n_games}", True, WHITE)
            text_score = font.render(f"Score: {self.score}", True, WHITE)

            self.display.blit(text_run_nb, (chart_x, 10))
            self.display.blit(text_score, (chart_x, 30))

            return

        recent_scores = self._plot_scores[-100:]
        recent_means = self._plot_mean_scores[-100:]

        local_max = max(max(recent_scores), max(recent_means))
        if local_max == 0: local_max = 1

        data_len = len(recent_scores)
        x_step = chart_w / (data_len - 1) if data_len > 1 else chart_w

        def _get_pt(i, val):
            # X = start + (index * step_size)
            px = chart_x + (i * x_step)

            # Y = bottom - ( value / local_max * height )
            py = (chart_y + chart_h) - (val / local_max * chart_h)
            return (int(px), int(py))

        if len(recent_means) > 1:
            points_mean = [_get_pt(i, m) for i, m in enumerate(recent_means)]
            pygame.draw.lines(self.display, (255, 0, 255), False, points_mean, 2)

        # Current Score / Record / Avg
        text_run_nb = font.render(f"Nb run: {self._n_games}", True, WHITE)
        text_score = font.render(f"Score: {self.score}", True, WHITE)
        text_rec = font.render(f"High score: {max(self._plot_scores)}", True, WHITE) # Global Record
        text_avg = font.render(f"Avg: {recent_means[-1]:.2f}", True, (255, 0, 255))

        self.display.blit(text_run_nb, (chart_x, 10))
        self.display.blit(text_score, (chart_x, 30))
        self.display.blit(text_rec, (chart_x + 150, 30))
        self.display.blit(text_avg, (chart_x, chart_y + chart_h + 30))

    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # [       0,    0,     0]
        # [straight, left, right]
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            raise Exception(f"invalid action {action}")

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
