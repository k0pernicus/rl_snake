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

N_HIDDEN_LAYERS = 256 # Size of the hidden layer in the QNet

DECAY_RATE = 0.995
MIN_EPSILON = 0.01

class SnakeAgent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0 # Control the randomness
        self.gamma   = DISCOUNT_RATE # Discount rate (must be smaller than 1)
        self.memory  = deque(maxlen=MAX_MEM) # Maximum memory of actions, rewards, ...
        self.model   = Linear_QNet(13, N_HIDDEN_LAYERS, 3) # 13 states, 3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_action(self, state):
        # random move : exploration vs exploitation
        # Let a (in theory) maximum of 1 game / 100 for exploration, even after a long time
        self.epsilon = max(self.epsilon * DECAY_RATE, MIN_EPSILON)
        moves = [0, 0, 0]

        # Important, as if the epsilon is too low then there will be no exploration,
        # and the snake will fall into local optimum problem
        if random.random() < self.epsilon:
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

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Standard vectors
        vec_l = (-BLOCK_SIZE, 0)
        vec_r = (BLOCK_SIZE, 0)
        vec_u = (0, -BLOCK_SIZE)
        vec_d = (0, BLOCK_SIZE)

        dirs = []
        if   dir_r: dirs = [vec_r, vec_d, vec_u, (BLOCK_SIZE, BLOCK_SIZE), (BLOCK_SIZE, -BLOCK_SIZE)]
        elif dir_l: dirs = [vec_l, vec_u, vec_d, (-BLOCK_SIZE, -BLOCK_SIZE), (-BLOCK_SIZE, BLOCK_SIZE)]
        elif dir_u: dirs = [vec_u, vec_r, vec_l, (BLOCK_SIZE, -BLOCK_SIZE), (-BLOCK_SIZE, -BLOCK_SIZE)]
        elif dir_d: dirs = [vec_d, vec_l, vec_r, (-BLOCK_SIZE, BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE)]

        def get_distance_from_collision(direction_vector):
            """
            Compute the distance between the current position of the snake and a collision
            This returns a float (good for the NN) between 0.0 (far away) and 1.0 (immediate danger)
            """
            current_pos = Point(head.x, head.y)
            dist = 0

            while True:
                current_pos = Point(current_pos.x + direction_vector[0],
                                    current_pos.y + direction_vector[1])
                dist += 1

                # If we hit something, return INVERSE distance
                if game._is_collision(current_pos):
                        # 1.0 = Immediate Death
                        # 0.5 = 1 step safety
                        # 0.1 = Far away
                    return 1.0 / dist

                # Optimization: Stop looking after 12 blocks to save performance
                if dist > 12: return 0.0

        collision_distance_ahead = get_distance_from_collision(dirs[0])
        collision_distance_right = get_distance_from_collision(dirs[1])
        collision_distance_left = get_distance_from_collision(dirs[2])
        collision_distance_diag_front_right = get_distance_from_collision(dirs[3])
        collision_distance_diag_front_left = get_distance_from_collision(dirs[4])

        state = [
            collision_distance_ahead,
            collision_distance_right,
            collision_distance_left,
            collision_distance_diag_front_right,
            collision_distance_diag_front_left,

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=float)

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
