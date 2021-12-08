from queue import Queue

MU = 1
INF = 100000
ESP = 1e-6


class NetworkFlow(object):
    def __init__(self, map_conf):
        self.n = map_conf['n_vertex']
        self.vertices = {}
        self.edges = {}
        self.g = []
        self.e = []
        self.m = 0

        for v in map_conf['vertices']:
            self.vertices[v['key']] = v
        for e in map_conf['edges']:
            self.edges[(e['start'], e['end'])] = e

    def create_edge(self, s, t, c, p):
        self.e.append(Edge(t, c, p))
        self.g[s].append(self.m)
        self.m += 1
        self.e.append(Edge(s, 0, -p))
        self.g[t].append(self.m)
        self.m += 1

    def min_cost_max_flow(self, s, t, n):
        while True:
            dist = [INF for _ in range(n)]
            pre = [-1 for _ in range(n)]
            flow = [0 for _ in range(n)]
            dist[s] = 0
            flow[s] = INF
            queue = Queue()
            queue.put(s)
            active = set()
            active.add(s)
            while not queue.empty():
                x = queue.get()
                active.remove(x)
                for i in self.g[x]:
                    if self.e[i].f < self.e[i].c and dist[x] + self.e[i].p < dist[self.e[i].t] - ESP:
                        y = self.e[i].t
                        dist[y] = dist[x] + self.e[i].p
                        pre[y] = i
                        flow[y] = min(flow[x], self.e[i].c - self.e[i].f)
                        if y not in active:
                            queue.put(y)
                            active.add(y)
            if dist[t] == INF:
                return
            x = t
            while x != s:
                i = pre[x]
                self.e[i].f += flow[t]
                i ^= 1
                self.e[i].f -= flow[t]
                x = self.e[i].t

    def choose_action(self, observation):
        self.g = [[] for _ in range(self.n + 2)]
        self.e = []
        self.m = 0
        st = observation['obs']
        for i in range(self.n):
            if st[i] > 0:
                self.create_edge(self.n, i, st[i], 0)
            elif st[i] < 0:
                self.create_edge(i, self.n + 1, -st[i], -MU)
        for i, vertex in self.vertices.items():
            self.create_edge(self.n, i, vertex['production'], 0)
            self.create_edge(i, self.n + 1, vertex['lambda'], -MU)
            self.create_edge(i, self.n + 1, vertex['upper_storage'], vertex['store_cost'])
            self.create_edge(i, self.n + 1, INF, vertex['loss_cost'])
        for vs, edge in self.edges.items():
            self.create_edge(vs[0], vs[1], edge['upper_capacity'], edge['trans_cost'] * edge['trans_time'])
        self.min_cost_max_flow(self.n, self.n + 1, self.n + 2)
        f = {}
        for edge in self.g[observation['controlled_player_index']]:
            f[self.e[edge].t] = self.e[edge].f
        action = []
        for index in observation['connected_player_index']:
            action.append(f[index])
        return action


class Edge:
    def __init__(self, t, c, p):
        self.t = t
        self.c = c
        self.p = p
        self.f = 0
