from spider.interface.highway_env import HighwayEnvBenchmark, HighwayEnvBenchmarkGUI
from spider.planner_zoo import LatticePlanner

from tqdm import tqdm


def test_benchmark_api():
    steps, dt = 20, 0.1

    benchmark = HighwayEnvBenchmark(dt=dt, config={"max_steps":200})


    planner = LatticePlanner({
        "steps": steps,
        'dt' : dt,
        "max_speed": 120/3.6,
        "end_s_candidates": (10,20,40,60),
        "end_l_candidates": (-4,0,4), # s,d采样生成横向轨迹 (-3.5, 0, 3.5), #
        "end_v_candidates": tuple(i*120/3.6/4 for i in range(5)), # 改这一项的时候，要连着限速一起改了
        "end_T_candidates": (1,2,4,8), # s_dot, T采样生成纵向轨迹
    })


    obs, info = benchmark.initial_environment()
    ego_veh_state, perception, local_map = benchmark.interface.wrap_observation(obs)
    benchmark.test(planner)

    # import cProfile
    # cProfile.run('for _ in range(10): output = planner.plan(ego_veh_state, perception, local_map)')
    # assert 0

    # cProfile.run('benchmark.test(planner)')


def test_benchmark_gui():
    import tkinter as tk
    from tkinter import ttk
    from spider.planner_zoo import LatticePlanner

    root = tk.Tk()
    app = HighwayEnvBenchmarkGUI(root)
    root.mainloop()

if __name__ == '__main__':
    test_benchmark_gui()


