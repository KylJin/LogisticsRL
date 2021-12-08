import os.path as osp
import pygame
import igraph
import numpy as np
from scipy.spatial import distance
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, rgb2hex

resource_path = osp.join(osp.dirname(__file__), 'resources')

# NOTE: FPS*SPD应为24的倍数，否则可能导致货车到达终点时偏移仓库图标中心
FPS = 60  # Frame Per Second，帧率，即每秒播放的帧数
SPD = 4  # Second Per Day，游戏中每天所占的秒数


class Truck(object):
    def __init__(self, start, end, trans_time, size=(32, 32)):
        self.image = pygame.image.load(osp.join(resource_path, "img/truck.png")).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)
        self.rect = self.image.get_rect()
        self.rect.center = start
        self.font = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)

        self.init_pos = (self.rect.x, self.rect.y)
        self.total_frame = trans_time * FPS * SPD // 24
        self.update_frame = 0

        speed_x = 24 * (end[0] - start[0]) / (trans_time * FPS * SPD)
        speed_y = 24 * (end[1] - start[1]) / (trans_time * FPS * SPD)
        self.speed = (speed_x, speed_y)

    def update(self):
        if self.update_frame < self.total_frame:
            self.update_frame += 1
            self.rect.x = self.init_pos[0] + self.speed[0] * self.update_frame
            self.rect.y = self.init_pos[1] + self.speed[1] * self.update_frame
        else:
            self.update_frame += 1
            if self.update_frame >= FPS * SPD:
                self.update_frame = 0
                self.rect.topleft = self.init_pos

    def draw(self, screen, action):
        if action <= 0:  # 若货车运输量为0，则不显示
            return
        # 当货车在道路上时才显示
        if 0 < self.update_frame < self.total_frame:
            screen.blit(self.image, self.rect)
            text = self.font.render(f"{round(action, 2)}", True, (44, 44, 44), (255, 255, 255))
            text_rect = text.get_rect()
            text_rect.centerx, text_rect.y = self.rect.centerx, self.rect.y - 12
            screen.blit(text, text_rect)


class LogisticsInterface(object):
    def __init__(self, width, height, network_data):
        self.width = width
        self.height = height
        self.v_radius = 40
        self.n_vertex = network_data['n_vertex']
        self.pd_gap = network_data['pd_gap']  # 每个节点生产量和平均消耗量之间的差距
        self.connections = network_data['connections']
        self.trans_times = network_data['trans_times']
        self.v_coords = self._spread_vertex(network_data['v_coords'])
        self.v_colors = []

        self.screen = pygame.display.set_mode([width, height])
        self.screen.fill("white")

        self.font1 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 24)
        self.font2 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 18)
        self.font3 = pygame.font.Font(osp.join(resource_path, "font/simhei.ttf"), 14)
        self.p_img = pygame.image.load(osp.join(resource_path, "img/produce.png")).convert_alpha()
        self.p_img = pygame.transform.scale(self.p_img, (16, 16))
        self.d_img = pygame.image.load(osp.join(resource_path, "img/demand.png")).convert_alpha()
        self.d_img = pygame.transform.scale(self.d_img, (14, 14))

        self.background = self.init_background()
        self.trucks = self.init_trucks()

    def init_background(self):
        # 绘制道路
        drawn_roads = []
        for i in range(self.n_vertex):
            start = self.v_coords[i]
            for j in self.connections[i]:
                if (j, i) in drawn_roads:
                    continue
                end = self.v_coords[j]
                self._rotated_road(start, end, width=12,
                                   border_color=(252, 122, 90), fill_color=(255, 172, 77))
                drawn_roads.append((i, j))

        # 绘制仓库节点
        norm = Normalize(vmin=min(self.pd_gap) - 2,
                         vmax=max(self.pd_gap) + 2)  # 数值映射范围（略微扩大）
        color_map = get_cmap('RdYlGn')  # 颜色映射表
        for coord, gap in zip(self.v_coords, self.pd_gap):
            rgb = color_map(norm(gap))[:3]
            color = pygame.Color(rgb2hex(rgb))
            light_color = self._lighten_color(color)
            pygame.draw.circle(self.screen, light_color, coord, self.v_radius, width=0)
            pygame.draw.circle(self.screen, color, coord, self.v_radius, width=2)
            self.v_colors.append(light_color)

        # 加入固定的提示
        self.add_notation()

        # 保存当前初始化的背景，便于后续刷新时使用
        background = self.screen.copy()
        return background

    @staticmethod
    def _lighten_color(color, alpha=0.1):
        r = alpha * color.r + (1 - alpha) * 255
        g = alpha * color.g + (1 - alpha) * 255
        b = alpha * color.b + (1 - alpha) * 255
        light_color = pygame.Color((r, g, b))
        return light_color

    def _spread_vertex(self, v_coords):
        if not v_coords:  # 若没有指定相对坐标，则随机将节点分布到画布上
            g = igraph.Graph()
            g.add_vertices(self.n_vertex)
            for i in range(self.n_vertex):
                for j in self.connections[i]:
                    g.add_edge(i, j)
            layout = g.layout_kamada_kawai()
            layout_coords = np.array(layout.coords).T
        else:  # 否则使用地图数据中指定的节点相对坐标
            layout_coords = np.array(v_coords).T

        # 将layout的坐标原点对齐到左上角
        layout_coords[0] = layout_coords[0] - layout_coords[0].min()
        layout_coords[1] = layout_coords[1] - layout_coords[1].min()

        # 将layout的坐标映射到画布坐标，并将图形整体居中
        stretch_rate = min((self.width - 2 * self.v_radius - 30) / layout_coords[0].max(),
                           (self.height - 2 * self.v_radius - 30) / layout_coords[1].max())
        margin_x = (self.width - layout_coords[0].max() * stretch_rate) // 2
        margin_y = (self.height - layout_coords[1].max() * stretch_rate) // 2
        vertex_coord = []
        for i in range(self.n_vertex):
            x = margin_x + int(layout_coords[0, i] * stretch_rate)
            y = margin_y + int(layout_coords[1, i] * stretch_rate)
            vertex_coord.append((x, y))

        return vertex_coord

    def _rotated_road(self, start, end, width, border_color=(0, 0, 0), fill_color=None):
        length = distance.euclidean(start, end)
        sin = (end[1] - start[1]) / length
        cos = (end[0] - start[0]) / length

        vertex = lambda e1, e2: (
            start[0] + (e1 * length * cos + e2 * width * sin) / 2,
            start[1] + (e1 * length * sin - e2 * width * cos) / 2
        )
        vertices = [vertex(*e) for e in [(0, -1), (0, 1), (2, 1), (2, -1)]]

        if not fill_color:
            pygame.draw.polygon(self.screen, border_color, vertices, width=3)
        else:
            pygame.draw.polygon(self.screen, fill_color, vertices, width=0)
            pygame.draw.polygon(self.screen, border_color, vertices, width=2)

    def init_trucks(self):
        trucks_list = []
        for i in range(self.n_vertex):
            start = self.v_coords[i]
            trucks = []
            for j, time in zip(self.connections[i], self.trans_times[i]):
                end = self.v_coords[j]
                truck = Truck(start, end, time)
                trucks.append(truck)
            trucks_list.append(trucks)

        return trucks_list

    def move_trucks(self, actions):
        for i in range(self.n_vertex):
            for truck, action in zip(self.trucks[i], actions[i]):
                truck.update()
                truck.draw(self.screen, action)

    def refresh_background(self, render_data):
        day = render_data['day']
        storages = render_data['storages']
        productions = render_data['productions']
        demands = render_data['demands']
        total_reward = render_data['total_reward']
        single_rewards = render_data['single_rewards']

        self.screen.blit(self.background, (0, 0))
        day_text = self.font1.render(f"第{day}天", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(day_text, (18, 10))
        r_text = self.font2.render(f"累计奖赏:{round(total_reward, 2)}", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(r_text, (18, 40))

        for coord, s, p, d, r, color in \
                zip(self.v_coords, storages, productions, demands, single_rewards, self.v_colors):
            s_text = self.font3.render(f"{round(s, 2)}", True, (44, 44, 44), color)
            s_text_rect = s_text.get_rect()
            s_text_rect.centerx, s_text_rect.y = coord[0], coord[1] - 32
            self.screen.blit(s_text, s_text_rect)

            p_text = self.font3.render(f"+{round(p, 2)}", True, (35, 138, 32), color)
            p_text_rect = p_text.get_rect()
            p_text_rect.centerx, p_text_rect.y = coord[0] + 9, coord[1] - 16
            self.screen.blit(p_text, p_text_rect)
            p_img_rect = self.p_img.get_rect()
            p_img_rect.centerx, p_img_rect.y = coord[0] - 13, coord[1] - 16
            self.screen.blit(self.p_img, p_img_rect)

            d_text = self.font3.render(f"-{round(d, 2)}", True, (251, 45, 45), color)
            d_text_rect = d_text.get_rect()
            d_text_rect.centerx, d_text_rect.y = coord[0] + 9, coord[1]
            self.screen.blit(d_text, d_text_rect)
            d_img_rect = self.d_img.get_rect()
            d_img_rect.centerx, d_img_rect.y = coord[0] - 13, coord[1]
            self.screen.blit(self.d_img, d_img_rect)

            r_text = self.font3.render(f"{round(r, 2)}", True, (12, 140, 210), color)
            r_text_rect = r_text.get_rect()
            r_text_rect.centerx, r_text_rect.y = coord[0], coord[1] + 16
            self.screen.blit(r_text, r_text_rect)

    def add_notation(self):
        text1 = self.font3.render("黑:库存量", True, (44, 44, 44), (255, 255, 255))
        self.screen.blit(text1, (18, 65))

        text2 = self.font3.render(":生产量", True, (35, 138, 32), (255, 255, 255))
        self.screen.blit(text2, (32, 85))
        self.screen.blit(self.p_img, (17, 85))

        text3 = self.font3.render(":消耗量", True, (251, 45, 45), (255, 255, 255))
        self.screen.blit(text3, (32, 105))
        self.screen.blit(self.d_img, (17, 105))

        text4 = self.font3.render("蓝:节点奖赏", True, (12, 140, 210), (255, 255, 255))
        self.screen.blit(text4, (18, 125))
