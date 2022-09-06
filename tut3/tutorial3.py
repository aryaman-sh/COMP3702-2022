import sys
import heapq
import time

"""
COMP3702 2021
Tutorial 3 Sample Solution

Last updated by njc 15/08/21
"""


class GridWorldEnv:

    # GridWorldState = (row, col) tuple

    ACTIONS = ['U', 'D', 'L', 'R']

    def __init__(self):
        self.n_rows = 9
        self.n_cols = 9

        # indexing is top to bottom, left to right (matrix indexing)
        init_r = 8
        init_c = 0
        self.init_state = (init_r, init_c)
        goal_r = 0
        goal_c = 8
        self.goal_state = (goal_r, goal_c)

        self.obstacles = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]

        self.costs = [[1, 1,  1,  5,  5,  5,  5, 1, 1],
                      [1, 1,  1,  5,  5,  5,  5, 1, 1],
                      [1, 1, 10, 10, 10, 10, 10, 1, 1],
                      [1, 1,  1, 10, 10, 10, 10, 1, 1],
                      [1, 1,  1,  1,  1, 10, 10, 1, 1],
                      [1, 1,  1,  1,  1, 10, 10, 1, 1],
                      [1, 1,  1,  1, 10, 10, 10, 1, 1],
                      [1, 1,  1, 10, 10, 10, 10, 1, 1],
                      [1, 1,  1,  1,  1,  1,  1, 1, 1]]

    def step(self, state, action):
        """
        :param state: (row, col) tuple
        :param action: 'U', 'D', 'L' or 'R'
        :return: (success [True/False], new state, action cost)
        """
        r, c = state

        if action == 'U':
            new_r = r - 1
            new_c = c
        elif action == 'D':
            new_r = r + 1
            new_c = c
        elif action == 'L':
            new_r = r
            new_c = c - 1
        elif action == 'R':
            new_r = r
            new_c = c + 1
        else:
            assert False, '!!! invalid action !!!'

        if (not (0 <= new_r < 9)) or (not (0 <= new_c < 9)) or self.obstacles[new_r][new_c] == 1:
            # collision occurs
            return False, (r, c), self.costs[r][c]
        else:
            return True, (new_r, new_c), self.costs[new_r][new_c]

    def is_goal(self, state):
        """
        :param state: (row, col) tuple
        :return: True/False
        """
        return state == self.goal_state

    def get_state_cost(self, state):
        r, c = state
        return self.costs[r][c]


class EightPuzzleState:

    def __init__(self, squares):
        if type(squares) is str:
            self.squares = list(squares)
        else:
            self.squares = [str(i) for i in squares]

        idx = -1
        for i in range(len(self.squares)):
            if self.squares[i] == '_':
                idx = i
        self.idx = idx

    def __eq__(self, obj):
        if obj is None:
            return False
        return tuple(self.squares) == tuple(obj.squares)

    def __hash__(self):
        return hash(tuple(self.squares))


class EightPuzzleEnv:

    ACTIONS = ['U', 'D', 'L', 'R']

    def __init__(self, init, goal):
        self.init_state = EightPuzzleState(init)
        self.goal_state = EightPuzzleState(goal)

    @staticmethod
    def move_left(state):
        new_squares = state.squares[:]
        new_squares[state.idx] = state.squares[state.idx - 1]
        new_squares[state.idx - 1] = state.squares[state.idx]
        return EightPuzzleState(new_squares)

    @staticmethod
    def move_right(state):
        new_squares = state.squares[:]
        new_squares[state.idx] = state.squares[state.idx + 1]
        new_squares[state.idx + 1] = state.squares[state.idx]
        return EightPuzzleState(new_squares)

    @staticmethod
    def move_up(state):
        new_squares = state.squares[:]
        new_squares[state.idx] = state.squares[state.idx - 3]
        new_squares[state.idx - 3] = state.squares[state.idx]
        return EightPuzzleState(new_squares)

    @staticmethod
    def move_down(state):
        new_squares = state.squares[:]
        new_squares[state.idx] = state.squares[state.idx + 3]
        new_squares[state.idx + 3] = state.squares[state.idx]
        return EightPuzzleState(new_squares)

    def step(self, state, action):
        """
        :param state: EightPuzzle state
        :param action: 'U', 'D', 'L' or 'R'
        :return: (success [True/False], new state, action cost)
        """
        if action == 'U' and (state.idx // 3) > 0:
            return True, self.move_up(state), 1
        elif action == 'D' and (state.idx // 3) < 2:
            return True, self.move_down(state), 1
        elif action == 'L' and (state.idx % 3) > 0:
            return True, self.move_left(state), 1
        elif action == 'R' and (state.idx % 3) < 2:
            return True, self.move_right(state), 1
        else:
            return False, EightPuzzleState(state.squares), 1

    def is_goal(self, state):
        """
        :param state: EightPuzzleState
        :return: True/False
        """
        return state == self.goal_state

    def get_state_cost(self, state):
        # same cost for all states in EightPuzzle
        return 1


class StateNode:

    def __init__(self, env, state, actions, path_cost):
        self.env = env
        self.state = state
        self.actions = actions
        self.path_cost = path_cost

    def get_successors(self):
        successors = []
        for a in GridWorldEnv.ACTIONS:
            success, new_state, a_cost = self.env.step(self.state, a)
            if success:
                successors.append(StateNode(self.env,
                                            new_state,
                                            self.actions + [a],
                                            self.path_cost + self.env.get_state_cost(new_state)))
        return successors

    def __lt__(self, other):
        # we won't use this as a priority directly, so result doesn't matter
        return True


def bfs(env, verbose=True):
    container = [StateNode(env, env.init_state, [], 0)]
    visited = set()

    n_expanded = 0
    while len(container) > 0:
        # expand node
        node = container.pop(0)

        # test for goal
        if env.is_goal(node.state):
            if verbose:
                print(f'Visited Nodes: {len(visited)},\t\tExpanded Nodes: {n_expanded},\t\t'
                      f'Nodes in Container: {len(container)}')
                print(f'Cost of Path (with Costly Moves): {node.path_cost}')
            return node.actions

        # add successors
        successors = node.get_successors()
        for s in successors:
            if s.state not in visited:
                container.append(s)
                visited.add(s.state)
        n_expanded += 1

    return None


def depth_limited_dfs(env, max_depth, verbose=True):
    container = [StateNode(env, env.init_state, [], 0)]
    # revisiting should be allowed if cost (depth) is lower than previous visit (needed for optimality)
    visited = {}    # dict mapping states to path cost (here equal to depth)

    n_expanded = 0
    while len(container) > 0:
        # expand node
        node = container.pop(-1)

        # test for goal
        if env.is_goal(node.state):
            if verbose:
                print(f'Visited Nodes: {len(visited.keys())},\t\tExpanded Nodes: {n_expanded},\t\t'
                      f'Nodes in Container: {len(container)}')
                print(f'Cost of Path (with Costly Moves): {node.path_cost}')
            return node.actions

        # add successors
        successors = node.get_successors()
        for s in successors:
            if (s.state not in visited or len(s.actions) < visited[s.state]) and len(s.actions) < max_depth:
                container.append(s)
                visited[s.state] = len(s.actions)
        n_expanded += 1

    return None


def iddfs(env, verbose=True):
    depth_limit = 1
    while depth_limit < 1000:
        actions = depth_limited_dfs(env, depth_limit, verbose)
        if actions is not None:
            return actions
        depth_limit += 1
    return None


def ucs(env, verbose=True):
    container = [(0, StateNode(env, env.init_state, [], 0))]
    heapq.heapify(container)
    # dict: state --> path_cost
    visited = {env.init_state: 0}
    n_expanded = 0
    while len(container) > 0:
        n_expanded += 1
        _, node = heapq.heappop(container)

        # check if this state is the goal
        if env.is_goal(node.state):
            if verbose:
                print(f'Visited Nodes: {len(visited.keys())},\t\tExpanded Nodes: {n_expanded},\t\t'
                      f'Nodes in Container: {len(container)}')
                print(f'Cost of Path (with Costly Moves): {node.path_cost}')
            return node.actions

        # add unvisited (or visited at higher path cost) successors to container
        successors = node.get_successors()
        for s in successors:
            if s.state not in visited.keys() or s.path_cost < visited[s.state]:
                visited[s.state] = s.path_cost
                heapq.heappush(container, (s.path_cost, s))

    return None


def manhattan_dist_heuristic(env, state):
    # Gridworld only
    return abs(env.goal_state[0] - state[0]) + abs(env.goal_state[1] - state[1])


def num_mismatches_heuristic(env, state):
    # EightPuzzle only
    mismatches = 0
    for tile in list('12345678_'):
        cur_idx = state.squares.index(tile)
        goal_idx = env.goal_state.squares.index(tile)
        if (cur_idx // 3) != (goal_idx // 3):
            mismatches += 1
        if (cur_idx % 3) != (goal_idx % 3):
            mismatches += 1
    return mismatches // 2


def summed_manhattan_heuristic(env, state):
    # EightPuzzle only
    total_displacement = 0
    for tile in list('12345678_'):
        cur_idx = state.squares.index(tile)
        cur_row = cur_idx // 3
        cur_col = cur_idx % 3
        goal_idx = env.goal_state.squares.index(tile)
        goal_row = goal_idx // 3
        goal_col = goal_idx % 3
        total_displacement += (abs(goal_row - cur_row) + abs(goal_col - cur_col))
    return total_displacement // 2


def a_star(env, heuristic, verbose=True):
    container = [(0 + heuristic(env, env.init_state), StateNode(env, env.init_state, [], 0))]
    heapq.heapify(container)
    # dict: state --> path_cost
    visited = {env.init_state: 0}
    n_expanded = 0
    while len(container) > 0:
        n_expanded += 1
        _, node = heapq.heappop(container)

        # check if this state is the goal
        if env.is_goal(node.state):
            if verbose:
                print(f'Visited Nodes: {len(visited.keys())},\t\tExpanded Nodes: {n_expanded},\t\t'
                      f'Nodes in Container: {len(container)}')
                print(f'Cost of Path (with Costly Moves): {node.path_cost}')
            return node.actions

        # add unvisited (or visited at higher path cost) successors to container
        successors = node.get_successors()
        for s in successors:
            if s.state not in visited.keys() or s.path_cost < visited[s.state]:
                visited[s.state] = s.path_cost
                heapq.heappush(container, (s.path_cost + heuristic(env, s.state), s))

    return None


def main(arglist):
    n_trials = 100
    print('== Exercise 3.1 ==============================================================================')
    gridworld = GridWorldEnv()

    print('BFS:')
    t0 = time.time()
    for i in range(n_trials):
        actions_bfs = bfs(gridworld, verbose=(i == 0))
    t_bfs = (time.time() - t0) / n_trials
    print(f'Num Actions: {len(actions_bfs)},\t\tActions: {actions_bfs}')
    print(f'Time: {t_bfs}')
    print('\n')

    print('IDDFS:')
    t0 = time.time()
    for i in range(n_trials):
        actions_iddfs = iddfs(gridworld, verbose=(i == 0))
    t_iddfs = (time.time() - t0) / n_trials
    print(f'Num Actions: {len(actions_iddfs)},\t\tActions: {actions_iddfs}')
    print(f'Time: {t_iddfs}')
    print('\n')

    print('== Exercise 3.2 ==============================================================================')
    print('UCS:')
    t0 = time.time()
    for i in range(n_trials):
        actions_ucs = ucs(gridworld, verbose=(i == 0))
    t_ucs = (time.time() - t0) / n_trials
    print(f'Num Actions: {len(actions_ucs)},\t\tActions: {actions_ucs}')
    print(f'Time: {t_ucs}')
    print('\n')

    print('A*:')
    t0 = time.time()
    for i in range(n_trials):
        actions_a_star = a_star(gridworld, manhattan_dist_heuristic, verbose=(i == 0))
    t_a_star = (time.time() - t0) / n_trials
    print(f'Num Actions: {len(actions_a_star)},\t\tActions: {actions_a_star}')
    print(f'Time: {t_a_star}')
    print('\n')

    print('== Exercise 3.3 ==============================================================================')
    puzzle = EightPuzzleEnv('281463_75', '1238_4765')

    print('BFS:')
    t0 = time.time()
    for i in range(n_trials):
        actions_bfs = bfs(puzzle, verbose=(i == 0))
    t_bfs = (time.time() - t0) / n_trials
    print(f'Num Actions: {len(actions_bfs)},\t\tActions: {actions_bfs}')
    print(f'Time: {t_bfs}')
    print('\n')

    print('A* (num mismatches):')
    t0 = time.time()
    for i in range(n_trials):
        actions_a_star = a_star(puzzle, num_mismatches_heuristic, verbose=(i == 0))
    t_a_star = (time.time() - t0) / n_trials
    print(f'Num Actions: {len(actions_a_star)},\t\tActions: {actions_a_star}')
    print(f'Time: {t_a_star}')
    print('\n')

    print('A* (summed manhattan):')
    t0 = time.time()
    for i in range(n_trials):
        actions_a_star = a_star(puzzle, summed_manhattan_heuristic, verbose=(i == 0))
    t_a_star = (time.time() - t0) / n_trials
    print(f'Num Actions: {len(actions_a_star)},\t\tActions: {actions_a_star}')
    print(f'Time: {t_a_star}')
    print('\n')


def show_visited(env, visited):
    for r in range(9):
        line = ''
        for c in range(9):
            if env.obstacles[r][c] == 1:
                line += 'X'
            elif (r, c) in visited:
                line += '1'
            else:
                line += '0'
        print(line)


if __name__ == '__main__':
    main(sys.argv[1:])



