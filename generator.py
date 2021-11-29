from random import randint, sample

MIN_NUM_VERTEX = 4
MAX_NUM_VERTEX = 15

MIN_PRODUCTION = 10
MAX_PRODUCTION = 50

MIN_INIT_STORAGE = 10
MAX_INIT_STORAGE = 120

MIN_UPPER_STORAGE = 30
MAX_UPPER_STORAGE = 150

MIN_STORE_COST = 1
MAX_STORE_COST = 5

MIN_LOSS_COST = 1
MAX_LOSS_COST = 5

MIN_LAMBDA = 10
MAX_LAMBDA = 50

MIN_UPPER_CAPACITY = 8
MAX_UPPER_CAPACITY = 20

MIN_TRANS_TIME = 4
MAX_TRANS_TIME = 24

MIN_TRANS_COST = 1
MAX_TRANS_COST = 3


def generate_random_map():
    num_vertex = randint(4, MAX_NUM_VERTEX)

    vertices, edges, connections = [], [], []
    for v in range(num_vertex):
        vertex = {
            "key": v,
            "production": randint(MIN_PRODUCTION, MAX_PRODUCTION),
            "init_storage": randint(MIN_INIT_STORAGE, MAX_INIT_STORAGE),
            "upper_storage": randint(MIN_UPPER_STORAGE, MAX_UPPER_STORAGE),
            "store_cost": randint(MIN_STORE_COST, MAX_STORE_COST) / 10,
            "loss_cost": randint(MIN_LOSS_COST, MAX_LOSS_COST) / 10,
            "lambda": randint(MIN_LAMBDA, MAX_LAMBDA)
        }
        vertices.append(vertex)

    num_circle = randint(3, num_vertex)
    used_vertex = sample(list(range(num_vertex)), num_circle)
    for i in range(num_circle):
        edge = {
            "start": used_vertex[i],
            "end": used_vertex[(i + 1) % num_circle],
            "upper_capacity": randint(MIN_UPPER_CAPACITY, MAX_UPPER_CAPACITY),
            "trans_time": randint(MIN_TRANS_TIME, MAX_TRANS_TIME),
            "trans_cost": randint(MIN_TRANS_COST, MAX_TRANS_COST) / 10
        }
        edges.append(edge)

    for v in range(num_vertex):
        if v in used_vertex:
            continue

        in_num = randint(1, len(used_vertex) - 1)
        in_vertex = sample(used_vertex, in_num)
        for i in in_vertex:
            edge = {
                "start": i,
                "end": v,
                "upper_capacity": randint(MIN_UPPER_CAPACITY, MAX_UPPER_CAPACITY),
                "trans_time": randint(MIN_TRANS_TIME, MAX_TRANS_TIME),
                "trans_cost": randint(MIN_TRANS_COST, MAX_TRANS_COST) / 10
            }
            edges.append(edge)

        left_vertex = list(set(used_vertex).difference(set(in_vertex)))
        out_num = randint(1, len(used_vertex) - in_num)
        out_vertex = sample(left_vertex, out_num)
        for i in out_vertex:
            edge = {
                "start": v,
                "end": i,
                "upper_capacity": randint(MIN_UPPER_CAPACITY, MAX_UPPER_CAPACITY),
                "trans_time": randint(MIN_TRANS_TIME, MAX_TRANS_TIME),
                "trans_cost": randint(MIN_TRANS_COST, MAX_TRANS_COST) / 10
            }
            edges.append(edge)

        used_vertex.append(v)

    map_data = {
        "n_vertex": num_vertex,
        "vertices": vertices,
        "edges": edges
    }

    return map_data
