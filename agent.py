import torch
import random
import numpy as np

from model import Linear_QNet, QTrainer
from snake import SnakeGame, Direction, Point, BLOCK_SIZE
from collections import deque

MAX_MEM = 100_000 # 100_000 items maximum
BATCH_SIZE = 1_000
LR = 0.01 # Learning Rate
DISCOUNT_RATE = 0.9

N_HIDDEN_LAYERS = 128 # Number of hidden layers in the QNet

class SnakeAgent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Control the randomness
        self.gamma   = DISCOUNT_RATE # Discount rate (must be smaller than 1)
        self.memory  = deque(maxlen=MAX_MEM) # Maximum memory of actions, rewards, ...
        self.model   = Linear_QNet(11, N_HIDDEN_LAYERS, 3) # 11 states, 3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_action(self, state):
        # random move : exploration vs exploitation
        # Let the first 60 plays do exploration, and privilege exploitation
        # until the end of the experiments
        self.epsilon = 60 - self.n_games
        moves = [0, 0, 0]

        # 200 seems too large here to not explore...
        if random.randint(0, 100) < self.epsilon:
            # Exploration...
            idx = random.randint(0, 2)
            moves[idx] = 1
        else:
            # Exploitation
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # will execute 'forward'
            idx = torch.argmax(prediction).item()
            moves[idx] = 1

        return moves

    def get_state(self, game):
        head = game.snake[0]

        coord_l = Point(head.x - BLOCK_SIZE, head.y)
        coord_r = Point(head.x + BLOCK_SIZE, head.y)
        coord_u = Point(head.x, head.y - BLOCK_SIZE)
        coord_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game._is_collision(coord_r)) or
            (dir_l and game._is_collision(coord_l)) or
            (dir_u and game._is_collision(coord_u)) or
            (dir_d and game._is_collision(coord_d)),
            # Danger right
            (dir_u and game._is_collision(coord_r)) or
            (dir_d and game._is_collision(coord_l)) or
            (dir_l and game._is_collision(coord_u)) or
            (dir_r and game._is_collision(coord_d)),
            # Danger left
            (dir_d and game._is_collision(coord_r)) or
            (dir_u and game._is_collision(coord_l)) or
            (dir_r and game._is_collision(coord_u)) or
            (dir_l and game._is_collision(coord_d)),
            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food
            game.food.x < game.head.x, # Food is left ?
            game.food.x > game.head.x, # Food is right ?
            game.food.y < game.head.y, # Food is up ?
            game.food.y > game.head.y, # Food is down ?
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)


def train():
    scores       = []
    mean_scores  = []
    total_scores = 0
    max_score       = 0

    agent = SnakeAgent()
    game  = SnakeGame()

    while True:
        # get the states
        current_state = agent.get_state(game)
        move = agent.get_action(current_state)

        # perform the action
        reward, game_over, score = game.play_step(move)
        new_state = agent.get_state(game)

        # Now, train the short memory, and remember
        agent.train_short_memory(current_state, move, reward, new_state, game_over)
        agent.remember(current_state, move, reward, new_state, game_over)

        if game_over:
            # Game Over -> Reset everything
            game.reset()
            agent.n_games += 1

            # In this case, replay memory in order to improve the agent
            agent.train_long_memory()

            if score > max_score:
                max_score = score
                agent.model.save()

            print(f"Game {agent.n_games}, score {score} with max score of {max_score}")

            scores.append(score)
            total_scores += score
            mean_score = total_scores / agent.n_games
            mean_scores.append(mean_score)

            # Let PyGame draw the stats for us
            game.update_plots(agent.n_games, scores, mean_scores)


if __name__ == "__main__":
    # Train the snake agent
    train()
