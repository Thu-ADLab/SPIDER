from spider.interface.BaseBenchmark import DummyBenchmark
from spider.planner_zoo import *

benchmark = DummyBenchmark({
    "save_video": True,
    "debug_mode": True
})

planner = FallbackDummyPlanner({
    "steps": 20,
    "dt": 0.2,
    "print_info": False
})

# planner = LatticePlanner({
#     "steps": 20,
#     "dt": 0.2,
# })

# planner = DummyPlanner()

# planner = BezierPlanner({
#     "steps": 20,
#     "dt": 0.2,
#     "end_s_candidates": (20,30),
#     "end_l_candidates": (-3.5, 0, 3.5),
#     "end_v_candidates": tuple(index * 60 / 3.6 / 3 for index in range(4)),  # 改这一项的时候，要连着限速一起改了
#     "end_T_candidates": (2, 4, 8),  # s_dot, T采样生成纵向轨迹
# })

# planner = PiecewiseLatticePlanner({
#     "steps": 20,
#     "dt": 0.2,
# })

# planner = ImaginaryPlanner()

# planner = OptimizedLatticePlanner({})

# planner = FallbackPlanner()


benchmark.test(planner)

