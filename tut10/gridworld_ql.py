import time
import math
import random

# Directions and Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}


# The map
OBSTACLES = [(1, 1)]
EXIT_STATE = (-1, -1)
REWARDS = {(3, 1): -100, (3, 2): 1}


class Grid:

    def __init__(self):
        self.x_size = 4
        self.y_size = 3
        self.p = 0.8
        self.discount = 0.9

        self.states = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2)]

    def attempt_move(self, s, a):
        """
        s: (x, y), x = s[0], y[1]
        Returns next state under deterministic action.
        """
        x, y = s[0], s[1]

        # Check absorbing state
        if s in REWARDS:
            return EXIT_STATE

        if s == EXIT_STATE:
            return s

        # Default: no movement
        result = s

        # Check borders:
        """
        Write code here to check if applying an action 
        keeps the agent with the boundary
        """
        if a == LEFT and x > 0:
            result = (x - 1, y)
        if a == RIGHT and x < self.x_size - 1:
            result = (x + 1, y)
        if a == UP and y < self.y_size - 1:
            result = (x, y + 1)
        if a == DOWN and y > 0:
            result = (x, y - 1)

        # Check obstacle cells
        if result in OBSTACLES:
            return s

        return result

    def stoch_action(self, a):
        # Stochastic actions probability distributions
        if a == RIGHT:
            return {RIGHT: self.p, UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        elif a == UP:
            return {UP: self.p, LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        elif a == LEFT:
            return {LEFT: self.p, UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        elif a == DOWN:
            return {DOWN: self.p, LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}

    def apply_move(self, s, a):
        # handle special cases
        if s in REWARDS.keys():
            # go to the exit state
            next_state = EXIT_STATE
            reward = REWARDS[s]
            return next_state, reward
        elif s == EXIT_STATE:
            # go to a new random state
            next_state = random.choice(self.states)
            reward = 0
            return next_state, reward

        # choose a random true action
        r = random.random()
        cumulative_prob = 0
        action = None
        for k, v in self.stoch_action(a).items():
            cumulative_prob += v
            if r < cumulative_prob:
                action = k
                break

        # apply true action
        next_state = self.attempt_move(s, action)
        reward = 0
        return next_state, reward


class QLearningAgent:

    EPSILON = 0.4
    ALPHA = 0.1

    def __init__(self, grid):
        self.grid = grid
        self.persistent_state = random.choice(grid.states)      # internal state used for training
        self.q_values = {}  # dict mapping (state, action) to float

    def next_iteration(self):
        # ===== select an action to perform (epsilon greedy exploration) =====
        best_q = -math.inf
        best_a = None
        for a in ACTIONS:
            if ((self.persistent_state, a) in self.q_values.keys() and
                    self.q_values[(self.persistent_state, a)] > best_q):
                best_q = self.q_values[(self.persistent_state, a)]
                best_a = a

        # epsilon chance to choose random action
        if best_a is None or random.random() < self.EPSILON:
            action = random.choice(ACTIONS)
        else:
            action = best_a

        # ===== simulate result of action =====
        next_state, reward = self.grid.apply_move(self.persistent_state, action)

        # ===== update value table =====
        # Q(s,a) <-- Q(s,a) + alpha * (temporal difference)
        # Q(s,a) <-- Q(s,a) + alpha * (target - Q(s, a))
        # target = r + gamma * max_{a' in A} Q(s', a')
        # compute target
        best_q1 = -math.inf
        best_a1 = None
        for a1 in ACTIONS:
            if ((next_state, a1) in self.q_values.keys() and
                    self.q_values[(next_state, a1)] > best_q1):
                best_q1 = self.q_values[(next_state, a1)]
                best_a1 = a1
        if best_a1 is None or next_state == EXIT_STATE:
            best_q1 = 0
        target = reward + (self.grid.discount * best_q1)
        if (self.persistent_state, action) in self.q_values:
            old_q = self.q_values[(self.persistent_state, action)]
        else:
            old_q = 0
        self.q_values[(self.persistent_state, action)] = old_q + (self.ALPHA * (target - old_q))

        # move to next state
        self.persistent_state = next_state

    def run_training(self, max_iterations):
        t0 = time.time()
        for i in range(max_iterations):
            self.next_iteration()
        print(f'Completed {max_iterations} iterations of training in {round(time.time() - t0, 1)} seconds.')

    def select_action(self, state):
        # choose the action with the highest Q-value for the given state
        best_q = -math.inf
        best_a = None
        for a in ACTIONS:
            if ((state, a) in self.q_values.keys() and
                    self.q_values[(state, a)] > best_q):
                best_q = self.q_values[(state, a)]
                best_a = a

        if best_a is None:
            return random.choice(ACTIONS)
        else:
            return best_a

    def print_values_and_policy(self):
        values = [[0, 0, 0, 0], [0, 'N/A', 0, 0], [0, 0, 0, 0]]
        policy = [['_', '_', '_', '_'], ['_', 'N/A', '_', '_'], ['_', '_', '_', '_']]
        for state in self.grid.states:
            best_q = -math.inf
            best_a = None
            for a in ACTIONS:
                if ((state, a) in self.q_values.keys() and
                        self.q_values[(state, a)] > best_q):
                    best_q = self.q_values[(state, a)]
                    best_a = a
            x, y = state
            values[y][x] = best_q
            policy[y][x] = best_a
        print('========== Values ==========')
        for row in reversed(values):
            line = '['
            for i, v in enumerate(row):
                if v != 'N/A':
                    line += str(round(v, 3))
                else:
                    line += 'N/A '
                if i != 3:
                    line += ', '
            line += ']'
            print(line)
        print('')
        print('========== Policy ==========')
        for row in reversed(policy):
            line = '['
            for i, p in enumerate(row):
                if p != 'N/A':
                    line += ' ' + ACTIONS_NAMES[p] + ' '
                else:
                    line += 'N/A'
                if i != 3:
                    line += ', '
            line += ']'
            print(line)


def main():
    env = Grid()
    agent = QLearningAgent(env)
    agent.run_training(10000)

    agent.print_values_and_policy()


if __name__ == "__main__":
    main()
