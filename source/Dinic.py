from collections import deque
from utils.args import args

class Dinic:
    def __init__(self, ps, es, n):
        self.n = n
        self.points = ps
        self.edges, self.cnt = [], 0
        self.head = [-1] * self.n
        self.deep = [0] * self.n
        self.now = [0] * self.n
        self.vis = [False] * self.n     # 仅用来dfs染色
        for e in es:
            self.add_edge(e[0], e[1], e[2])

    def add_edge(self, u, v, w):
        self.edges.append(
            {
                'v': v,
                'w': w,
                'next': self.head[u]
            }
        )
        self.head[u] = self.cnt
        self.cnt += 1
        self.edges.append(
            {
                'v': u,
                'w': w,
                'next': self.head[v]
            }
        )
        self.head[v] = self.cnt
        self.cnt += 1

    def bfs(self, s, t):
        self.deep = [0] * self.n
        q = deque()
        q.append(s)
        self.now[s] = self.head[s]
        self.deep[s] = 1
        while len(q):
            u = q.popleft()
            i = self.head[u]
            while i != -1:
                w, v = self.edges[i]['w'], self.edges[i]['v']
                if not (self.deep[v] or not w):
                    self.deep[v] = self.deep[u] + 1
                    self.now[v] = self.head[v]
                    q.append(v)
                    if v == t:
                        return True
                i = self.edges[i]['next']
        return False

    def dinic(self, s, t, flow):
        if s == t:
            return flow
        rest = flow
        i = self.now[s]
        while i != -1 and rest:
            w, v = self.edges[i]['w'], self.edges[i]['v']
            if w and self.deep[v] == self.deep[s] + 1:
                deep_rest = self.dinic(v, t, min(rest, w))
                if not deep_rest:
                    self.deep[v] = 0
                self.edges[i]['w'] -= deep_rest
                self.edges[i ^ 1]['w'] += deep_rest
                rest -= deep_rest
            i = self.edges[i]['next']
        self.now[s] = i
        return flow - rest

    def solve(self, s, t):
        ans = 0
        while self.bfs(s, t):
            flow = 1
            while flow:
                flow = self.dinic(s, t, args.maxValidFloat)
                ans += flow
        return ans

    def dye(self, s):
        q = deque()
        q.append(s)
        self.vis[s] = True
        while len(q):
            u = q.popleft()
            i = self.head[u]
            while i != -1:
                v = self.edges[i]['v']
                if not self.vis[v]:
                    if self.edges[i]['w']:
                        q.append(v)
                        self.vis[v] = True
                i = self.edges[i]['next']



    def get_foreground(self, s):
        self.dye(s)
        foreground = []
        for i, v in enumerate(self.vis):
            if v and i != s:
                foreground.append(i)
        return foreground

    def get_background(self, t):
        background = []
        for i, v in enumerate(self.vis):
            if not v and i != t:
                background.append(i)
        return background

