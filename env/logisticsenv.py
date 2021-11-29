from env.simulators.game import Game
from env.obs_interfaces.observation import DictObservation
import numpy as np
from utils.box import Box


class LogisticsEnv(Game, DictObservation):
    def __init__(self, conf, map_conf):
        super().__init__(map_conf['n_vertex'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], map_conf['n_vertex'], conf['obs_type'])
        self.map_conf = map_conf
        self.max_step = int(conf['max_step'])
        self.step_cnt = 0

        self.players = []
        self.current_state = self.init_map()
        self.all_observes = self.get_all_observations()
        # 每个玩家的action space list, 可以根据player_id获取对应的single_action_space
        self.joint_action_space = self.set_action_space()
        self.info = {
            'upper_storages': [self.players[i].upper_storage for i in range(self.n_player)],
            'upper_capacity': [act.high.tolist() for act in self.joint_action_space]
        }

    def init_map(self):
        # 添加图中的节点
        vertices = self.map_conf['vertices'].copy()
        for vertex_info in vertices:
            key = vertex_info['key']
            self.add_vertex(key, vertex_info)

        # 添加图中的有向边
        edges = self.map_conf['edges'].copy()
        for edge_info in edges:
            start = edge_info['start']
            end = edge_info['end']
            self.add_edge(start, end, edge_info)

        # 对每个节点进行初始化
        init_state = []
        for i in range(self.n_player):
            self.players[i].update_init_storage()
            init_state.append(self.players[i].init_storage)

        return init_state

    def add_vertex(self, key, vertex_info):
        vertex = LogisticsVertex(key, vertex_info)
        self.players.append(vertex)

    def add_edge(self, start, end, edge_info):
        edge = LogisticsEdge(edge_info)
        start_vertex = self.players[start]
        start_vertex.add_neighbor(end, edge)

    def reset(self):
        self.step_cnt = 0
        self.players = []
        self.current_state = self.init_map()
        self.all_observes = self.get_all_observations()

        return self.all_observes

    def step(self, all_actions):
        self.step_cnt += 1
        all_actions = self.bound_actions(all_actions)
        self.info.update({'actual_actions': all_actions})

        self.current_state = self.get_next_state(all_actions)
        self.all_observes = self.get_all_observations()

        reward, single_rewards = self.get_reward(all_actions)
        done = self.is_terminal()
        self.info.update({
            'reward': reward,
            'single_rewards': single_rewards
        })

        return self.all_observes, reward, done, self.info

    def bound_actions(self, all_actions):  # 对每个节点的动作进行约束
        bounded_actions = []

        for i in range(self.n_player):
            vertex = self.players[i]
            action = all_actions[i].copy()
            actual_trans = sum(action)
            if vertex.init_storage < 0:  # 初始库存量为负（有货物缺口）
                bounded_actions.append([0] * len(action))
            elif actual_trans > vertex.init_storage:  # 运出的总货物量超过初始库存量
                # 每条运输途径的货物量进行等比例缩放
                bounded_action = [act * vertex.init_storage / actual_trans for act in action]
                bounded_actions.append(bounded_action)
            else:  # 合法动作
                bounded_actions.append(action)

        return bounded_actions

    def get_next_state(self, all_actions):
        assert len(all_actions) == self.n_player
        # 统计每个节点当天运出的货物量out_storages，以及接收的货物量in_storages
        out_storages, in_storages = [0] * self.n_player, [0] * self.n_player
        for i in range(self.n_player):
            action = all_actions[i]
            out_storages[i] = sum(action)
            connections = self.players[i].get_connections()
            for (act, nbr) in zip(action, connections):
                in_storages[nbr] += act

        # 更新每个节点当天的最终库存量以及下一天的初始库存量，
        # 并记录每个节点当天最开始的初始库存start_storages、生产量productions和消耗量demands，用于可视化
        next_state = []
        start_storages, productions, demands = [], [], []
        for i in range(self.n_player):
            start_storages.append(self.players[i].final_storage)
            productions.append(self.players[i].production)
            demands.append(self.players[i].demand)
            self.players[i].update_final_storage(out_storages[i], in_storages[i])
            self.players[i].update_init_storage()
            next_state.append(self.players[i].init_storage)
        self.info.update({
            'start_storages': start_storages,
            'productions': productions,
            'demands': demands
        })

        return next_state

    def get_dict_observation(self, current_state, player_id, info_before):
        obs = {
            "obs": current_state,
            "connected_player_index": self.players[player_id].get_connections(),
            "controlled_player_index": player_id
        }
        return obs

    def get_all_observations(self, info_before=''):
        all_obs = self.get_dict_many_observation(
            self.current_state,
            range(self.n_player),
            info_before
        )
        return all_obs

    def get_reward(self, all_actions):
        total_reward = 0
        single_rewards = []
        for i in range(self.n_player):
            action = all_actions[i]
            reward = self.players[i].calc_reward(action)
            total_reward += reward
            single_rewards.append(reward)

        return total_reward, single_rewards

    def set_action_space(self):
        action_space = []

        for i in range(self.n_player):
            vertex = self.players[i]
            high = []
            for j in vertex.get_connections():
                edge = vertex.get_edge(j)
                high.append(edge.upper_capacity)
            action_space_i = Box(np.zeros(len(high)), np.array(high), dtype=np.float64)
            action_space.append(action_space_i)

        return action_space

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_single_connections(self, player_id):
        return self.players[player_id].get_connections()

    def is_terminal(self):
        is_done = self.step_cnt >= self.max_step
        return is_done

    def get_network_data(self):
        network_data = {
            'n_vertex': self.n_player,
            'v_coords': self.map_conf.get('coords')
        }

        pd_gap, edges, edges_length = [], [], []
        for i in range(self.n_player):
            vertex = self.players[i]
            pd_gap.append(vertex.production - vertex.lambda_)
            for j in vertex.get_connections():
                edges.append((i, j))
                edges_length.append(vertex.get_edge(j).trans_time)
        network_data['pd_gap'] = pd_gap  # 记录每个节点生产量和平均消耗量之间的差距
        network_data['roads'] = edges
        network_data['roads_length'] = edges_length

        return network_data

    def get_render_data(self, current_state=None):
        render_data = {
            'day': self.step_cnt,
            'storages': self.info['start_storages'],
            'productions': self.info['productions'],
            'demands': self.info['demands'],
            'reward': self.info['reward']
        }

        actions = []
        for action_i in self.info['actual_actions']:
            actions += action_i
        render_data['actions'] = actions

        return render_data


class LogisticsVertex(object):
    def __init__(self, key, info):
        self.key = key
        self.connectedTo = {}

        self.production = info['production']
        self.init_storage = 0
        self.final_storage = info['init_storage']
        self.upper_storage = info['upper_storage']
        self.store_cost = info['store_cost']
        self.loss_cost = info['loss_cost']
        self.storage_loss = 0  # 更新完当天的最终库存量后，统计当天的库存溢出量
        self.init_storage_loss = 0  # 因为每次状态更新会提前计算下一天的初始库存量，
        # 若不单独记录初始库存的溢出量，则会在计算每日reward时出错
        self.lambda_ = info['lambda']
        self.demand = 0

    def add_neighbor(self, nbr, edge):
        self.connectedTo.update({nbr: edge})

    def get_connections(self):
        return list(self.connectedTo.keys())

    def get_edge(self, nbr):
        return self.connectedTo.get(nbr)

    def get_demand(self):
        demand = np.random.poisson(lam=self.lambda_, size=1)
        return demand[0]

    def update_init_storage(self):
        self.demand = self.get_demand()
        self.init_storage = self.final_storage - self.demand + self.production
        self.init_storage_loss = 0
        if self.init_storage > self.upper_storage:  # 当天初始库存量超过存储上限
            self.init_storage_loss = self.init_storage - self.upper_storage
            self.init_storage = self.upper_storage

    def update_final_storage(self, out_storage, in_storage):
        self.final_storage = self.init_storage - out_storage + in_storage
        self.storage_loss = self.init_storage_loss
        if self.final_storage > self.upper_storage:  # 当天最终库存量超过存储上限
            self.storage_loss += (self.final_storage - self.upper_storage)
            self.final_storage = self.upper_storage

    def calc_reward(self, action, mu=10):
        connections = self.get_connections()
        assert len(action) == len(connections)
        # 舍弃超过库存货物造成的损失
        reward = -self.loss_cost * self.storage_loss

        # 当日运输货物的成本
        for (act, nbr) in zip(action, connections):
            edge = self.get_edge(nbr)
            reward -= (edge.trans_cost * edge.trans_time * act)

        if self.final_storage >= 0:  # 因库存盈余所导致的存储成本
            reward -= (self.store_cost * self.final_storage)
        else:  # 因库存空缺而加的惩罚项
            reward += (mu * self.final_storage)

        return reward


class LogisticsEdge(object):
    def __init__(self, info):
        self.upper_capacity = info['upper_capacity']
        self.trans_time = info['trans_time']
        self.trans_cost = info['trans_cost']
