import os
import sys
import json
import argparse
import pygame
from env import LogisticsEnv
from agent.Random_Agent import RandomAgent
from agent.Rule_Agent import RuleAgent
from interface.logistics_interface import LogisticsInterface, FPS, SPD
from generator import generate_random_map

AgentDict = {
    'random': RandomAgent,
    'rule': RuleAgent
}


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


def store_map_config(config):
    maps_path = os.path.join(os.path.dirname(__file__), "map")
    count = 0
    while True:
        new_file = os.path.join(maps_path, f"map_{count}.json")
        if os.path.exists(new_file):
            count += 1
        else:
            break
    with open(new_file, 'w') as f:
        json.dump(config, f)


def run_game(args):
    # 读取配置文件
    all_conf = load_config("env/config.json")
    conf = all_conf['Logistics_Transportation']
    if args.map == -1:
        map_conf = generate_random_map()
        if args.store_map:
            store_map_config(map_conf)
    else:
        map_conf = load_config(f"map/map_{args.map}.json")

    # 根据配置文件创建环境
    env = LogisticsEnv(conf, map_conf)
    network_data = env.get_network_data()

    # 创建智能体
    n_player = network_data['n_vertex']
    Agent = AgentDict[args.algo]
    agents = []
    for i in range(n_player):
        connections = env.get_single_connections(i)
        action_space = env.get_single_action_space(i)
        agent = Agent(key=i,
                      connections=connections,
                      n_actions=action_space.shape[0],
                      low=action_space.low,
                      high=action_space.high)
        agents.append(agent)

    # 可视化界面初始化
    pygame.init()
    pygame.display.set_caption("Simple Logistics Simulator")
    interface_ctrl = LogisticsInterface(args.width, args.height, network_data)

    observation = env.reset()
    FPSClock = pygame.time.Clock()
    # 开始游戏
    while not env.is_terminal():
        for _ in range(args.step_per_update):
            joint_actions = []
            for idx in range(n_player):
                observation_i = observation[idx]['obs']
                action_i = agents[idx].choose_action(observation_i)
                joint_actions.append(action_i)
            observation, reward, done, info = env.step(joint_actions)

        if not args.silence:
            print("start_storages:", info['start_storages'])
            print("productions:", info['productions'])
            print("demands:", info['demands'])
            print("upper_storages:", info['upper_storages'])
            print()
            print("observation:", observation[0]['obs'])
            print("actual_actions:", info['actual_actions'])
            print("upper_capacity:", info['upper_capacity'])
            print()
            print("reward:", reward)
            print("single_rewards:", info['single_rewards'])
            print('-------------------------------------')

        render_data = env.get_render_data()

        # 绘制当前step的动画
        current_frame = 0
        while current_frame < FPS * SPD:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            interface_ctrl.refresh_background(render_data)
            interface_ctrl.move_trucks(render_data['actions'])
            current_frame += 1

            FPSClock.tick(FPS)
            pygame.display.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', default=1000, type=int)
    parser.add_argument('--height', default=800, type=int)
    parser.add_argument('--map', default=1, type=int)  # 指定地图编号，若为-1则随机生成
    parser.add_argument('--store_map', action='store_true')  # 是否保存随机生成的地图
    parser.add_argument('--algo', default="rule", type=str, help="random/rule")
    parser.add_argument('--step_per_update', default=100, type=int)  # 两次更新的step间隔
    parser.add_argument('--silence', action='store_true')  # 添加为True，即不在控制台打印结果；不加为False
    args = parser.parse_args()

    run_game(args)


if __name__ == '__main__':
    main()
