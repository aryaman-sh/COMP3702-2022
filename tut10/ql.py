"""
Q-Learning

1) Initialise Q(s,a) arbitrarily
2) observe current state s
3) Repeat for each episode until convergence
    select and carry out action a
    observer reward r and state s'
    evalue new Q(s,a)
    s <-- s'
4) For each state s
    get best policy
"""

class QLearningAgent:

    def __init__(self, grid):
        """
        Initialise a grid
        """
        self.grid = grid
        
        # random state
        self.persistent_state = random.choice(grid.states)      # internal state used for training
        
        # q values table, similar to async VI
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
        if best_a is None or random.random() < self.EPSILON: # some epsilon for choosing best action
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
