import random
import math
import tkinter as tk

UP = 'U'
DOWN = 'D'
LEFT = 'L'
RIGHT = 'R'

ACTIONS = [UP, DOWN, LEFT, RIGHT]

ACTION_DELTAS = {
	UP: (-1, 0),
	DOWN: (1, 0),
	LEFT: (0, -1),
	RIGHT: (0, 1)
}

VALUE_FONT = ('Arial', 24)
POLICY_FONT = ('Arial', 16)

POLICY_CHARS = {
    UP: '↑',
    DOWN: '↓',
    LEFT: '←',
    RIGHT: '→'
}

class GridWorld:
	def __init__(
		self,
		num_rows: int,
		num_columns: int,
		obstacles: list[tuple[int, int]],
		terminal_states: list[tuple[int, int]],
		rewards: dict[tuple[int, int], float],
		transition_probability: float,
		discount: float
	):
		self.num_rows = num_rows
		self.num_cols = num_columns
		self.obstacles = obstacles
		self.states = [(x, y) for x in range(self.num_rows) for y in range(self.num_cols) if (x, y) not in obstacles]
		self.discount = discount
		self.terminal_states = terminal_states
		self.rewards = rewards
		self.p = transition_probability


	def attempt_move(self, state: tuple[int, int], action: str) -> tuple[int, int]:
		if state in self.terminal_states:
			return state

		x, y = state
		dx, dy = ACTION_DELTAS.get(action)
		new_state = max(min(x + dx, self.num_rows - 1), 0), max(min(y + dy, self.num_cols - 1), 0)

		if new_state in self.obstacles:
			return state

		return new_state
		
	def stoch_action(self, a: str) -> dict[str, float]:
		# Stochastic actions probability distributions
		if a == RIGHT: 
			return {RIGHT: 0.8, UP: 0.1, DOWN: 0.1}
		elif a == UP:
			return {UP: 0.8, LEFT: 0.1, RIGHT: 0.1}
		elif a == LEFT:
			return {LEFT: 0.8, UP: 0.1, DOWN: 0.1}
		elif a == DOWN:
			return {DOWN: 0.8, LEFT: 0.1, RIGHT: 0.1}

	def perform_action(self, state: tuple[int, int], action: str) -> tuple[int, int]:
		actions = list(self.stoch_action(action).items())
		action_chosen = random.choices([item[0] for item in actions], weights=[item[1] for item in actions])[0]
		next_state = self.attempt_move(state, action_chosen)
		return next_state

	def get_reward(self, state: tuple[int, int]) -> float:
		return self.rewards.get(state, 0.0)

class MCTS:
	# This class is adapted from the A2 sample solution with credit to Nick Collins
	VISITS_PER_SIM = 1
	MAX_ROLLOUT_DEPTH = 200
	TRIALS_PER_ROLLOUT = 1
	EXP_BIAS = 4000 # This is set very high for demo so can easily predict next selected

	def __init__(self, env):
		self.env = env
		self.q_sa = {}
		self.n_s = {}
		self.n_sa = {}

	def selection(self, state):
		""" Given a state, selects the next action based on UCB1. """
		unvisited = []
		for a in ACTIONS:
			if (state, a) not in self.n_sa:
				unvisited.append(a)
		if unvisited:
			# If theres an unvisited, go there to see what it's like
			return random.choice(unvisited)

		# They've all been visited, so pick which one to try again based on UCB1
		best_u = -float('inf')
		best_a = None
		for a in ACTIONS:
			u = self.q_sa.get((state, a), 0) + (self.EXP_BIAS * math.sqrt(math.log(self.n_s.get(state, 0))/self.n_sa.get((state, a), 1)))
			if u > best_u:
				best_u = u
				best_a = a
		return best_a if best_a is not None else random.choice(ACTIONS)

	def simulate(self, initial_state):
		# self.initial_state = initial_state
		visited = {}
		return self.mcts_search(initial_state, 0, visited)

	def plan_online(self, state, iters=10000):
		max_iter = iters
		for i in range(max_iter):
			self.simulate(state)
		return self.mcts_select_action(state)

	def mcts_search(self, state, depth, visited):
		# Check for non-visit conditions
		if (state in visited and visited[state] >= self.VISITS_PER_SIM) or (depth > self.MAX_ROLLOUT_DEPTH):
			# Choose the best Q-value if one exists
			best_q = -float('inf')
			best_a = None
			for a in ACTIONS:
				if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
					best_q = self.q_sa[(state, a)]
					best_a = a
			if best_a is not None:
				return best_q
			else:
				return self.mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
		else:
			visited[state] = visited.get(state, 0) + 1

		# Check for terminal state
		if state in self.env.terminal_states:
			self.n_s[state] = 1
			return self.env.get_reward(state)

		# Check for leaf node:
		if state not in self.n_s:
			# Reached an unexpanded state (i.e. simulation time) so perform rollout from here
			self.n_s[state] = 0
			return self.mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
		else:
			action = self.selection(state)

			# Update counts
			self.n_sa[(state, action)] = self.n_sa.get((state, action), 0) + 1
			self.n_s[state] += 1

			# Execute the selected action and recurse
			new_state = self.env.perform_action(state, action)
			r = self.env.get_reward(new_state) + self.env.discount * self.mcts_search(new_state, depth+1, visited)

			# update node statistics
			if (state, action) not in self.q_sa:
				self.q_sa[(state, action)] = r
			else:
				self.q_sa[(state, action)] = ((self.q_sa[(state, action)] * self.n_sa[(state, action)]) + r) / (self.n_sa[(state, action)] + 1)

			return r

	def mcts_random_rollout(self, state, max_depth, trials):
		total = 0
		s = state
		for i in range(trials):
			d = 0
			while d < max_depth and not s in self.env.terminal_states:
				action = random.choice(ACTIONS)
				new_state = self.env.perform_action(s, action)
				reward = self.env.get_reward(new_state)
				total += (self.env.discount ** (d+1)) * (reward)
				s = new_state
				d += 1
		return total / trials

	def mcts_select_action(self, state):
		best_q = -float('inf')
		best_a = None
		for a in ACTIONS:
			if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
				best_q = self.q_sa[(state, a)]
				best_a = a
		return best_a

	def extract_policy(self):
		policy = {}
		for row in range(self.env.num_rows):
			for col in range(self.env.num_cols):
				state = (row, col)
				action = self.mcts_select_action(state)
				policy[state] = action
		return policy

	def __str__(self):
		return str(self.q_sa) + ':' + str(self.n_s) + ':' + str(self.n_sa)

	def __repr__(self):
		return str(self)

class GridWorldView(tk.Canvas):
	BOARD_WIDTH = 1000
	BOARD_HEIGHT = 750
	POLICY_CHARS = {
		UP: '↑',
		DOWN: '↓',
		LEFT: '←',
		RIGHT: '→'
	}
	def __init__(self, master, num_rows, num_cols):
		super().__init__(master, width=self.BOARD_WIDTH, height=self.BOARD_HEIGHT)
		self._master = master
		self.num_rows = num_rows
		self.num_cols = num_cols

	def _get_cell_size(self) -> tuple[int, int]:
		return self.BOARD_WIDTH // self.num_cols, self.BOARD_HEIGHT // self.num_rows

	def _get_bbox(self, cell: tuple[int, int]) -> tuple[int, int, int, int]:
		row, col = cell
		cell_width, cell_height = self._get_cell_size()
		x_min, y_min = col * cell_width, row * cell_height
		x_max, y_max = x_min + cell_width, y_min + cell_height
		return x_min, y_min, x_max, y_max

	def _get_midpoint(self, cell: tuple[int, int]) -> tuple[int, int]:
		x_min, y_min, x_max, y_max = self._get_bbox(cell)
		return (x_min + x_max) // 2, (y_min + y_max) // 2

	def redraw(self, rewards, q_sa, n_sa, n_s, policy) -> None:
		self.clear()
		for row in range(self.num_rows):
			for col in range(self.num_cols):
				state = (row, col)
				self.draw_cell(state, rewards.get(state, 0), q_sa, n_sa, n_s, policy)

	def draw_cell(self, state, reward, q_sa, n_sa, n_s, policy):
		if n_s.get(state) is None:
			return

		colour = 'light grey'
		if reward:
			colour = 'light green' if reward > 0 else 'orange'
		self.create_rectangle(self._get_bbox(state), fill=colour)

		if reward:
			self.create_text(self._get_midpoint(state), text=str(reward), font=VALUE_FONT)
			return # In grid world all rewards are terminal states so don't need more info drawn

		# Draw lines to segment cell into 4 sections
		x_min, y_min, x_max, y_max = self._get_bbox(state)
		self.create_line((x_min, y_min), (x_max, y_max))
		self.create_line((x_max, y_min), (x_min, y_max))

		# Draw (state, action) info: TODO refactor; this is pretty bad duplication
		x, y = self._get_midpoint(state)
		BUFFER = self._get_cell_size()[0] // 3

		up_q = q_sa.get((state, UP))
		if up_q:
			up_q = round(up_q, 2)
			self.create_text(x, y-BUFFER, text=f'Q: {up_q}', font=POLICY_FONT)

		down_q = q_sa.get((state, DOWN))
		if down_q:
			down_q = round(down_q, 2)
			self.create_text(x, y+BUFFER, text=f'Q: {down_q}', font=POLICY_FONT)

		right_q = q_sa.get((state, RIGHT))
		if right_q:
			right_q = round(right_q, 2)
			self.create_text(x + BUFFER, y, text=f'Q: {right_q}', font=POLICY_FONT)

		left_q = q_sa.get((state, LEFT))
		if left_q:
			left_q = round(left_q, 2)
			self.create_text(x - BUFFER, y, text=f'Q: {left_q}', font=POLICY_FONT)

		BUFFER2 = 15

		up_n = n_sa.get((state, UP))
		if up_n:
			self.create_text(x, y-BUFFER - BUFFER2, text=f'N: {up_n}', font=POLICY_FONT)

		down_n = n_sa.get((state, DOWN))
		if down_n:
			self.create_text(x, y+BUFFER - BUFFER2, text=f'N:{down_n}', font=POLICY_FONT)

		right_n = n_sa.get((state, RIGHT))
		if right_n:
			self.create_text(x + BUFFER, y - BUFFER2, text=f'N: {right_n}', font=POLICY_FONT)

		left_n = n_sa.get((state, LEFT))
		if left_n:
			self.create_text(x - BUFFER, y - BUFFER2, text=f'N:{left_n}', font=POLICY_FONT)

		# Draw N(s) and the current policy
		self.create_rectangle((x - 40, y - 20), (x + 40, y + 20), fill='pink')
		self.create_text(x - 5, y, text=str(n_s.get(state, '')) + POLICY_CHARS.get(policy.get(state), ''), font=POLICY_FONT)

	def draw_current_cell(self, cell):
		# This method just puts a blue box around the cell the agent is in
		x_min, y_min, x_max, y_max = self._get_bbox(cell)
		self.create_line((x_min, y_min), (x_min, y_max), width=4, fill='blue')
		self.create_line((x_max, y_min), (x_max, y_max), width=4, fill='blue')
		self.create_line((x_min, y_min), (x_max, y_min), width=4, fill='blue')
		self.create_line((x_min, y_max), (x_max, y_max), width=4, fill='blue')

	def clear(self) -> None:
		""" Clears all items off this canvas. """
		self.delete(tk.ALL)

class MCTSView:
	def __init__(self, master, num_rows, num_cols, move_command, simulate_command):
		self.grid = GridWorldView(master, num_rows, num_cols)
		self.grid.pack(side=tk.LEFT)

		controls = tk.Frame(master)
		controls.pack(side=tk.LEFT)

		self._num_sims_label = tk.Label(controls, text='# Simulations')
		self._num_sims_label.pack()

		self._pos_label = tk.Label(controls, text='Current position:')
		self._pos_label.pack()

		self._current_move_label = tk.Label(controls, text='Current recommended move:')
		self._current_move_label.pack()

		btn_frame = tk.Frame(controls)
		btn_frame.pack()
		entry = tk.Entry(btn_frame)
		entry.pack(side=tk.LEFT)
		tk.Button(btn_frame, text='Simulate', command=lambda: simulate_command(int(entry.get()))).pack(side=tk.LEFT)

		move_frame = tk.Frame(controls)
		move_frame.pack()

		tk.Button(move_frame, text='UP', command=lambda : move_command(UP)).pack(side=tk.LEFT)
		tk.Button(move_frame, text='DOWN', command=lambda : move_command(DOWN)).pack(side=tk.LEFT)
		tk.Button(move_frame, text='LEFT', command=lambda : move_command(LEFT)).pack(side=tk.LEFT)
		tk.Button(move_frame, text='RIGHT', command=lambda : move_command(RIGHT)).pack(side=tk.LEFT)

	def redraw_grid(self, rewards, q_sa, n_sa, n_s, policy, current_state):
		self.grid.redraw(rewards, q_sa, n_sa, n_s, policy)
		self.grid.draw_current_cell(current_state)

	def update_labels(self, position, num_sims):
		self._pos_label.config(text=f'Current position: {position}')
		self._num_sims_label.config(text=f'# Simulations: {num_sims}')

class MCTSController:
	def __init__(self, master):
		num_rows = 3
		num_cols = 4
		obstacles = [(1, 1)]
		rewards = {
			(1, 3): -1,
			(0, 3): 1,
		}
		self.current_state = (2, 0)
		terminal_states = list(rewards.keys())
		self.grid = GridWorld(num_rows, num_cols, obstacles, terminal_states, rewards, 0.8, 0.9)
		
		INITIAL_STATE = (0, 0)
		self.mcts = MCTS(self.grid)
		self.mcts.plan_online(self.current_state, iters=1)

		mcts = self.mcts
		self._num_sims = 0
		self.view = MCTSView(master, num_rows, num_cols, self.move, self.sim)

		self.rewards = rewards
		self.view.redraw_grid(self.rewards, self.mcts.q_sa, self.mcts.n_sa, self.mcts.n_s, mcts.extract_policy(), self.current_state)
		self.view.update_labels(self.current_state, self._num_sims)

	def move(self, action):
		self.current_state = self.grid.perform_action(self.current_state, action)
		self.view.update_labels(self.current_state, self._num_sims)
		self.view.redraw_grid(self.rewards, self.mcts.q_sa, self.mcts.n_sa, self.mcts.n_s, self.mcts.extract_policy(), self.current_state)

	def sim(self, num_iter):
		self._num_sims += num_iter
		self.mcts.plan_online(self.current_state, iters=num_iter)
		self.view.redraw_grid(self.rewards, self.mcts.q_sa, self.mcts.n_sa, self.mcts.n_s, self.mcts.extract_policy(), self.current_state)
		self.view.update_labels(self.current_state, self._num_sims)

if __name__ == '__main__':
	root = tk.Tk()
	app = MCTSController(root)
	root.mainloop()
