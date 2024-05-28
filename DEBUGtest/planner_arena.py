from spider.interface.BaseBenchmark import DummyBenchmark
from spider.planner_zoo import *
from spider.planner_zoo.DQNPlanner import DQNPlanner
from spider.planner_zoo.DDQNPlanner import DDQNPlanner
from spider.planner_zoo.DiscretePPOPlanner import DiscretePPOPlanner
from spider.planner_zoo.ProbabilisticPlanner import ProbabilisticPlanner
from spider.planner_zoo.MlpPlanner import MlpPlanner
from spider.planner_zoo.GRUPlanner import GRUPlanner

planners = []

planner = FallbackDummyPlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})
planners.append(planner)

planner = LatticePlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})
planners.append(planner)

planner = OptimizedLatticePlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})
planners.append(planner)


planner = BezierPlanner({
    "steps": 20,
    "dt": 0.2,
    "end_s_candidates": (20,30),
    "end_l_candidates": (-3.5, 0, 3.5),
    "end_v_candidates": tuple(index * 60 / 3.6 / 3 for index in range(4)),  # 改这一项的时候，要连着限速一起改了
    "end_T_candidates": (2, 4, 8),  # s_dot, T采样生成纵向轨迹
    "print_info": False
})
planners.append(planner)

planner = PiecewiseLatticePlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})
planners.append(planner)

planner = ImaginaryPlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})
planners.append(planner)

planner = DQNPlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})
planner.policy.load_model('./RL_planner/q_net.pth')
planners.append(planner)

planner = DiscretePPOPlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})
planner.policy.load_model('./RL_planner/ppo.pth')
planners.append(planner)
#
#
# planner = DDQNPlanner({
#     "steps": 20,
#     "dt": 0.2,
#     "num_object": 5,
#     "print_info": False
# })
# planner.policy.load_model('./RL_planner/q_net_bes_ddqn.pth')
# planners.append(planner)

########## IL ##########
il_cfg = {
    "steps": 20,
    "dt": 0.2,
    "num_object": 5,
    "normalize": False,
    "relative": False,
    "longitudinal_range": (-50, 100),
    "lateral_range": (-20,20),

    "learning_rate": 0.0001,
    "enable_tensorboard": False,
}

planner = MlpPlanner(il_cfg)
planner.load_state_dict('./mlp.pth')
planners.append(planner)

planner = GRUPlanner(il_cfg)
planner.load_state_dict('./gru.pth')
planners.append(planner)

planner = ProbabilisticPlanner(il_cfg)
planner.load_state_dict('./prob.pth')
planners.append(planner)


benchmark = DummyBenchmark({
    # "save_video": True,
    # "debug_mode": True,
    "snapshot": False,
    "evaluation": True,
    "rendering": False,
})

import random

for planner in planners:
    random.seed(0)

    print("--------------------------------------")
    print("Planner: ", planner.__class__.__name__)
    benchmark.test(planner, 10)
