from copy import deepcopy
from spider.elements import TrackingBoxList, Trajectory

class ImaginaryEngine:
    def __init__(self, steps, dt, atom_planner, predictor=None, tracker=None, track_steps=5):

        assert steps%track_steps==0
        assert dt == atom_planner.dt, 'imagining intervals and atom planner intervals should be the same.'


        self.dt = dt
        self.steps = steps
        self.dream_times = steps//track_steps

        self.atom_planner = atom_planner
        self.delta_steps = track_steps # 每一次做梦执行多少步atom_planner规划出来的轨迹

        # 目前就是线性预测器
        self.predictor = predictor

        # 目前不加轨迹跟踪器, 假设完美跟踪
        self.tracker = tracker


    def imagine(self, ego_veh_state, obstacles:TrackingBoxList, local_map):
        assert isinstance(obstacles, TrackingBoxList)
        ego_veh_state = deepcopy(ego_veh_state)
        obstacles = deepcopy(obstacles)

        idx = self.delta_steps
        tbox_list = obstacles
        imaginary_trajectory = Trajectory(self.steps, self.dt)
        truncated = False

        for i in range(self.dream_times):

            # planning
            atom_traj:Trajectory = self.atom_planner.plan(ego_veh_state, obstacles, local_map)

            if atom_traj is None:
                print("No feasible trajectory! Imagination End.")
                truncated = True
                break

            # control: track the trajectory for track_steps
            if self.tracker is None:
                ego_veh_state = ego_veh_state.from_traj_step(atom_traj, idx)
            else:
                raise NotImplementedError("control module not implemented yet.")

            # predict
            if self.predictor is None:
                # update other vehicle
                for tb in tbox_list:
                    xc, yc, length, width, heading = tb.obb
                    vx, vy = tb.vx, tb.vy
                    tb.set_obb(
                        [xc + vx * atom_traj.dt * idx, yc + vy * atom_traj.dt * idx, length, width, heading])
            else:
                raise NotImplementedError("prediction module not implemented yet.")

            # truncate and concat trajectory
            imaginary_trajectory.concat(atom_traj.truncate(steps_num=idx))

        return imaginary_trajectory, truncated



class DoctorStrange(ImaginaryEngine):
    def __init__(self, steps, dt, atom_planner, predictor=None, tracker=None, track_steps=5):
        print("\033[1;32m"
              "Congrats, Dr. Strange! You have got the Eye Of Agamotto.\n"
              "The power of peeking into the future is at hand..."
              "\033[0m")
        super(DoctorStrange, self).__init__(steps, dt, atom_planner, predictor, tracker, track_steps)


if __name__ == '__main__':
    from spider.planner_zoo import LatticePlanner
    from spider.interface.BaseBenchmark import DummyBenchmark

    planner = LatticePlanner({
        "steps": 10,
        "dt": 0.2,
        "end_s_candidates": (20, 40, 60),
        "end_l_candidates": (-3.5, 0, 3.5),
        "print_info": False
    })
    track_steps = 2
    engine = DoctorStrange(30, 0.2, planner, track_steps=track_steps)

    ego_state, obstacles, local_map = DummyBenchmark.get_environment_presets()
    ego_state.velocity.x = 1.0
    ego_state.calc_kinematics()

    traj, truncated = engine.imagine(ego_state, obstacles, local_map)

    import spider.visualize as vis
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    def draw():
        for lane in local_map.lanes:
            plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1.5)  # 画地图

        for tb in obstacles:
            vis.draw_boundingbox(tb, color='black', fill=True, alpha=0.1, linestyle='-', linewidth=1.5)  # 画他车
            # 画他车预测轨迹
            tb_pred_traj = np.column_stack((tb.x + traj.t * tb.vx, tb.y + traj.t * tb.vy))
            vis.draw_polyline(tb_pred_traj, show_buffer=True, buffer_dist=tb.width * 0.5, buffer_alpha=0.1,
                              color='C3')

        vis.draw_ego_history(ego_state, '-', lw=1, color='gray')  # 画自车历史
        vis.draw_trajectory(traj, '.-', show_footprint=True, color='C2')  # 画轨迹
        if "control_points" in traj.debug_info:
            pts = traj.debug_info["control_points"]
            plt.plot(pts[:, 0], pts[:, 1], 'or')
        vis.draw_ego_vehicle(ego_state, color='C0', fill=True, alpha=0.3, linestyle='-', linewidth=1.5)  # 画自车
        vis.ego_centric_view(ego_state.x(), ego_state.y(), [-20, 80], [-5, 5])

    vis.adjust_canvas()
    # 可视化
    while ego_state.x() < 240:
        traj, truncated = engine.imagine(ego_state, obstacles, local_map)
        plt.cla()
        draw()
        plt.pause(0.001)

        ego_state = ego_state.from_traj_step(traj, 1) # 执行轨迹

        # 更新了obstacles的位置
        for tb in obstacles:
            tb.set_obb([tb.x + tb.vx * traj.dt, tb.y + tb.vy * traj.dt, tb.length, tb.width, tb.box_heading])

    plt.show()



