import time
import numpy as np
import random

# Directions and Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}


def get_action_name(action):
    return ACTIONS_NAMES[action]


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

        # self.states = list((x, y) for x in range(self.x_size) for y in range(self.y_size))
        # self.states.append(EXIT_STATE)
        # for obstacle in OBSTACLES:
        #     self.states.remove(obstacle)

        self.states = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2),
                       EXIT_STATE]

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

    def get_reward(self, s):
        if s == EXIT_STATE:
            return 0

        if s in REWARDS:
            return REWARDS[s]
        else:
            return 0


class ValueIteration:
    def __init__(self, grid):
        self.grid = Grid()
        self.values = {state: 0 for state in self.grid.states}
        self.policy = {state: RIGHT for state in self.grid.states}
        self.epsilon = 0.001
        self.converged = False

    def next_iteration(self):
        """
        Write code here to imlpement the VI value update
        Iterate over self.grid.states and ACTIONS
        Use stoch_action(a) and attempt_move(s,a)
        """
        new_values = dict()
        new_policy = dict()
        for s in self.grid.states:
            # Keep track of maximum value
            action_values = dict()
            for a in ACTIONS:
                total = 0
                for stoch_action, p in self.grid.stoch_action(a).items():
                    # Apply action
                    s_next = self.grid.attempt_move(s, stoch_action)
                    total += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
                action_values[a] = total
            # Update state value with best action
            new_values[s] = max(action_values.values())
            new_policy[s] = dict_argmax(action_values)


        # Check convergence
        differences = [abs(self.values[s] - new_values[s]) for s in self.grid.states]
        if max(differences) < self.epsilon:
            self.converged = True

        # Update values
        self.values = new_values
        self.policy = new_policy

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)
        print("Converged:", self.converged)

    def print_values_and_policy(self):
        for state in self.grid.states:
            print(state, get_action_name(self.policy[state]), self.values[state])
        print("Converged:", self.converged)


def dict_argmax(d):
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k


class PolicyIteration:
    def __init__(self, grid):
        self.grid = Grid()
        self.values = {state: 0 for state in self.grid.states}
        self.policy = {state: RIGHT for state in self.grid.states}
        self.converged = False
        self.epsilon = 0.001
        self.USE_LIN_ALG = True

        # t model (lin alg)
        t_model = np.zeros([len(self.grid.states), len(ACTIONS), len(self.grid.states)])
        for i, s in enumerate(self.grid.states):
            for j, a in enumerate(ACTIONS):
                if s in REWARDS.keys():
                    for k in range(len(self.grid.states)):
                        if self.grid.states[k] == (-1, -1):
                            t_model[i][j][k] = 1.0
                        else:
                            t_model[i][j][k] = 0.0
                elif s == EXIT_STATE:
                    t_model[i][j][self.grid.states.index(EXIT_STATE)] = 1.0
                else:
                    for stoch_action, p in self.grid.stoch_action(a).items():
                        # Apply action
                        s_next = self.grid.attempt_move(s, stoch_action)
                        k = self.grid.states.index(s_next)
                        t_model[i][j][k] += p
        self.t_model = t_model

        # r model (lin alg)
        r_model = np.zeros([len(self.grid.states)])
        for i, s in enumerate(self.grid.states):
            r_model[i] = self.grid.get_reward(s)
        self.r_model = r_model

        # lin alg policy
        la_policy = np.zeros([len(self.grid.states)], dtype=np.int64)
        for i, s in enumerate(self.grid.states):
            la_policy[i] = 3
            # la_policy[i] = random.randint(0, len(ACTIONS) - 1)
        self.la_policy = la_policy

    def next_iteration(self):
        new_values = dict()
        new_policy = dict()

        # policy evaluation
        if not self.USE_LIN_ALG:
            # use 'naive'/iterative policy evaluation
            value_converged = False
            while not value_converged:
                new_values = dict()
                for s in self.grid.states:
                    total = 0
                    for stoch_action, p in self.grid.stoch_action(self.policy[s]).items():
                        # Apply action
                        s_next = self.grid.attempt_move(s, stoch_action)
                        total += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
                    # Update state value with best action
                    new_values[s] = total

                # Check convergence
                differences = [abs(self.values[s] - new_values[s]) for s in self.grid.states]
                if max(differences) < self.epsilon:
                    value_converged = True

                # Update values and policy
                self.values = new_values
        else:
            # use linear algebra for policy evaluation
            # V^pi = R + gamma T^pi V^pi
            # (I - gamma * T^pi) V^pi = R
            # Ax = b; A = (I - gamma * T^pi),  b = R
            state_numbers = np.array(range(len(self.grid.states)))  # indices of every state
            t_pi = self.t_model[state_numbers, self.la_policy]
            values = np.linalg.solve(np.identity(len(self.grid.states)) - (self.grid.discount * t_pi), self.r_model)
            self.values = {s: values[i] for i, s in enumerate(self.grid.states)}
            new_policy = {s: ACTIONS[self.la_policy[i]] for i, s in enumerate(self.grid.states)}

        # policy improvement
        for s in self.grid.states:
            # Keep track of maximum value
            action_values = dict()
            for a in ACTIONS:
                total = 0
                for stoch_action, p in self.grid.stoch_action(a).items():
                    # Apply action
                    s_next = self.grid.attempt_move(s, stoch_action)
                    total += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
                action_values[a] = total
            # Update policy
            new_policy[s] = dict_argmax(action_values)

        # Check convergence
        if new_policy == self.policy:
            self.converged = True

        self.policy = new_policy
        if self.USE_LIN_ALG:
            for i, s in enumerate(self.grid.states):
                self.la_policy[i] = self.policy[s]

    def print_values_and_policy(self):
        for state in self.grid.states:
            print(state, get_action_name(self.policy[state]), self.values[state])
        print("Converged:", self.converged)


def run_value_iteration():
    grid = Grid
    vi = ValueIteration(grid)

    start = time.time()
    print("Initial values:")
    vi.print_values()
    print()

    max_iter = 30

    for i in range(max_iter):
        vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values_and_policy()
        print()
        if vi.converged:
            break

    end = time.time()
    print("Time to complete", i + 1, "VI iterations")
    print(end - start)


def run_policy_iteration():
    grid = Grid
    pi = PolicyIteration(grid)

    start = time.time()
    print("Initial policy and values:")
    pi.print_values_and_policy()
    print()

    max_iter = 50

    for i in range(max_iter):
        pi.next_iteration()
        print("Policy and values after iteration", i + 1)
        pi.print_values_and_policy()
        print()
        if pi.converged:
            break

    end = time.time()
    print("Time to complete", i + 1, "PI iterations")
    print(end - start)


if __name__ == "__main__":
    mode = input("Please enter 1 for value iteration, or 2 for policy iteration: ").strip()
    if mode == "1":
        run_value_iteration()
    elif mode == "2":
        run_policy_iteration()
    else:
        print("Invalid option.")
