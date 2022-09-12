import copy
import numpy as np
import random
import time

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

OBSTACLES = [(1, 1)]
EXIT_STATE = (-1, -1)

MAX_ITER = 1000
EPSILON = 0.0001

class Grid:
    def __init__(self):
        self.x_size = 4
        self.y_size = 3
        self.p = 0.8
        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.rewards = {(3, 1): -100, (3, 2): 1}
        self.discount = 0.9

        self.states = list((x, y) for x in range(self.x_size) for y in range(self.y_size))
        self.states.append(EXIT_STATE)
        for obstacle in OBSTACLES:
            self.states.remove(obstacle)

    def attempt_move(self, s, a):
        """ Attempts to move the agent from state s via action a.

            Parameters:
                s: The current state.
                a: The *actual* action performed (as opposed to the chosen
                   action; i.e. you do not need to account for non-determinism
                   in this method).
            Returns: the state resulting from performing action a in state s.
        """
        x, y = s

        # Check absorbing state
        if s == EXIT_STATE:
            return s
        if s in self.rewards.keys():
            return EXIT_STATE

        # Default: no movement
        result = s 

        # Check borders
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
        """ Returns the probabilities with which each action will actually occur,
            given that action a was requested.

        Parameters:
            a: The action requested by the agent.

        Returns:
            The probability distribution over actual actions that may occur.
        """
        if a == RIGHT:
            return {RIGHT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        elif a == UP:
            return {UP: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        elif a == LEFT:
            return {LEFT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        return {DOWN: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}

    def get_transition_probabilities(self, s, a):
        """ Calculates the probability distribution over next states given
            action a is taken in state s.

        Parameters:
            s: The state the agent is in
            a: The action requested

        Returns:
            A map from the reachable next states to the probabilities of reaching
            those state; i.e. each item in the returned dictionary is of form
            s' : P(s'|s,a)
        """
        probabilities = {}
        for action, probability in self.stoch_action(a).items():
            next_state = self.attempt_move(s, action)
            probabilities[next_state] = probabilities.get(next_state, 0) + probability
        return probabilities

    def get_reward(self, s):
        """ Returns the reward for being in state s. """
        if s == EXIT_STATE:
            return 0

        return self.rewards.get(s, 0)

def dict_argmax(d):
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k

class ValueIteration:
    def __init__(self, grid):
        self.grid = grid
        self.values = {state: 0 for state in self.grid.states}
        self.policy = {state: RIGHT for state in self.grid.states}
        self.converged = False
        self.differences = []

    def next_iteration(self):
        new_values = dict()
        new_policy = dict()
        for s in self.grid.states:
            # Keep track of maximum value
            action_values = dict()
            for a in self.grid.actions:
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
        max_diff = max(differences)
        self.differences.append(max_diff)

        if max_diff < EPSILON:
            self.converged = True

        # Update values
        self.values = new_values
        self.policy = new_policy

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)

    def print_policy(self):
        for state, action in self.policy.items():
            print(state, ACTION_NAMES[action])


class PolicyIteration:
    def __init__(self, grid):
        self.grid = grid
        self.values = {state: 0 for state in self.grid.states}
        self.policy = {pi: RIGHT for pi in self.grid.states}
        self.r = [0 for s in self.grid.states]
        self.USE_LIN_ALG = False
        self.converged = False

        # Full transition matrix (P) of dimensionality |S|x|A|x|S| since its
        # not specific to any one policy. We'll slice out a |S|x|S| matrix
        # from it for each policy evaluation
        # t model (lin alg)
        self.t_model = np.zeros([len(self.grid.states), len(self.grid.actions), len(self.grid.states)])
        for i, s in enumerate(self.grid.states):
            for j, a in enumerate(self.grid.actions):
                transitions = self.grid.get_transition_probabilities(s, a)
                for next_state, prob in transitions.items():
                    self.t_model[i][j][self.grid.states.index(next_state)] = prob

        # Reward vector
        r_model = np.zeros([len(self.grid.states)])
        for i, s in enumerate(self.grid.states):
            r_model[i] = self.grid.get_reward(s)
        self.r_model = r_model
    
       # lin alg policy
        la_policy = np.zeros([len(self.grid.states)], dtype=np.int64)
        for i, s in enumerate(self.grid.states):
            la_policy[i] = 3 # Allocate arbitrary initial policy
        self.la_policy = la_policy

    def next_iteration(self):
        new_values = dict()
        new_policy = dict()

        self.policy_evaluation()
        new_policy = self.policy_improvement()
        self.convergence_check(new_policy)

    def policy_evaluation(self):
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
                if max(differences) < EPSILON:
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
            
        # return new_policy

    def policy_improvement(self):
        if self.USE_LIN_ALG:
            new_policy = {s: self.grid.actions[self.la_policy[i]] for i, s in enumerate(self.grid.states)}
        else:
            new_policy = {}

        for s in self.grid.states:
            # Keep track of maximum value
            action_values = dict()
            for a in self.grid.actions:
                total = 0
                for stoch_action, p in self.grid.stoch_action(a).items():
                    # Apply action
                    s_next = self.grid.attempt_move(s, stoch_action)
                    total += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
                action_values[a] = total
            # Update policy
            new_policy[s] = dict_argmax(action_values)
        return new_policy

    def convergence_check(self, new_policy):
        if new_policy == self.policy:
            self.converged = True

        self.policy = new_policy
        if self.USE_LIN_ALG:
            for i, s in enumerate(self.grid.states):
                self.la_policy[i] = self.policy[s]
        
    def print_values(self):
        for state, value in self.values.items():
            print(state, value)
    
    def print_policy(self):
        for state, policy in self.policy.items():
            print(state, policy)

def plot_vi_diffs(diffs):
    import matplotlib.pyplot as plt
    # Plot from iteration 2 onwards to make trend clearer
    xs = range(2, len(diffs) + 1)
    plt.plot(xs, diffs[1:])
    plt.xlabel('# iterations')
    plt.ylabel('Max. difference')
    plt.show()

def run_vi():
    grid = Grid()
    vi = ValueIteration(grid)

    start = time.time()
    print("Initial values:")
    vi.print_values()
    print()

    for i in range(MAX_ITER):
        vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values()
        print("Policy after iteration", i + 1)
        vi.print_policy()
        print()
        if vi.converged:
            break

    end = time.time()
    print("Time to complete", i + 1, "VI iterations")
    print(end - start)

def run_pi():
    grid = Grid()
    pi = PolicyIteration(grid)

    start = time.time()
    print("Initial values:")
    pi.print_values()
    print()

    for i in range(MAX_ITER):
        pi.next_iteration()
        print("Values after iteration", i + 1)
        pi.print_values()
        print("Policy after iteration", i + 1)
        # pi.print_policy()
        print()
        if pi.converged:
            break

    

    end = time.time()
    print("Time to complete", i + 1, "VI iterations")
    print(end - start)

if __name__ == "__main__":
    run_vi()
    