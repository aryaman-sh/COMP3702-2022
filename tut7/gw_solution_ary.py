import copy
import numpy as np
import random
import time

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}

def get_action_name(action):
    return ACTIONS_NAMES[action]

OBSTACLES = [(1, 1)]
EXIT_STATE = (-1, -1)

MAX_ITER = 100
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

        # Default: no movement
        result = s

        # Check borders
        """
        TODO: Write code here to check if applying an action
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
        """
        TODO: Write code here to check if applying an action
        moves the agent into an obstacle cell
        """
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
        """
            TODO: Create and return a dictionary mapping each possible next state to the
            probability that that state will be reached by doing a in s.
        """

    def get_reward(self, s):
        """ Returns the reward for being in state s. """
        if s == EXIT_STATE:
            return 0

        return self.rewards.get(s, 0)

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


class PolicyIteration:
    def __init__(self, grid):
        self.grid = grid
        self.values = {state: 0 for state in self.grid.states}
        self.policy = {pi: RIGHT for pi in self.grid.states}
        self.r = [0 for s in self.grid.states]
        for idx, state in enumerate(self.grid.states):
            if state in self.grid.rewards.keys():
                self.r[idx] = self.grid.rewards[state]
        print('r is ', self.r)

    def next_iteration(self):
        """
        TODO: Write code to orchestrate one iteration of PI here.
        """
        return

    def policy_evaluation(self):
        """
        TODO: Write code for the policy evaluation step of PI here. That is, update
        the current value estimates using the current policy estimate.
        """
        return

    def policy_improvement(self):
        """
        TODO: Write code to extract the best policy for a given value function here
        """
        return

    def convergence_check(self):
        """
        TODO: Write code to check if PI has converged here
        """
        return

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)

    def print_policy(self):
        for state, policy in self.policy.items():
            print(state, policy)

def dict_argmax(d):
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k

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


if __name__ == "__main__":
    mode = input("Please enter 1 for value iteration, or 2 for policy iteration: ").strip()
    if mode == "1":
        run_value_iteration()
    elif mode == "2":
        run_policy_iteration()
    else:
        print("Invalid option.")
