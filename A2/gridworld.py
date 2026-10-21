ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
ARROWS = ["^", ">", "v", "<"]


class GridWorld:
    def __init__(self, rows, cols, start, terminals, cliff=(),
                 step_reward=-1.0, cliff_reward=-100.0):
        self.rows, self.cols = rows, cols
        self.n_states = rows * cols
        self.n_actions = len(ACTIONS)
        self.start = self._idx(start)
        self.terminals = {self._idx(t) for t in terminals}
        self.cliff = {self._idx(c) for c in cliff}
        self.step_reward, self.cliff_reward = step_reward, cliff_reward
        self.state = self.start
        self.P = [[self._transition(s, a) for a in range(self.n_actions)]
                  for s in range(self.n_states)]

    def _idx(self, rc):
        return rc[0] * self.cols + rc[1]

    def _transition(self, s, a):
        if s in self.terminals:
            return s, 0.0, True
        r, c = divmod(s, self.cols)
        dr, dc = ACTIONS[a]
        nr = min(max(r + dr, 0), self.rows - 1)
        nc = min(max(c + dc, 0), self.cols - 1)
        s2 = nr * self.cols + nc
        if s2 in self.cliff:
            return self.start, self.cliff_reward, False
        return s2, self.step_reward, s2 in self.terminals

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        s2, r, done = self.P[self.state][action]
        self.state = s2
        return s2, r, done, {}


def four_by_four():
    return GridWorld(4, 4, start=(3, 0), terminals=[(0, 0), (3, 3)])


def cliff_world():
    return GridWorld(4, 12, start=(3, 0), terminals=[(3, 11)],
                     cliff=[(3, c) for c in range(1, 11)])