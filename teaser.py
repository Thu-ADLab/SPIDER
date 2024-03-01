# import matplotlib.pyplot as plt
# import numpy as np
#
# from spider.planner_zoo.LatticePlanner import LatticePlanner
#
# from spider.elements.map import RoutedLocalMap
# from spider.elements.Box import TrackingBoxList, TrackingBox
# from spider.elements.map import Lane
# from spider.elements.vehicle import VehicleState, Transform, Location, Rotation,Vector3D
# import spider.visualize as vis


__all__ = ['demo', 'teaser']

def teaser():
    return demo() # 执行spider.teaser(), 就会执行demo()

def demo():
    from spider.interface.BaseBenchmark import DummyBenchmark
    from spider.planner_zoo import LatticePlanner

    planner = LatticePlanner({
        "steps": 15,
        "dt": 0.2,
        "end_s_candidates": (20, 40, 60),
        "end_l_candidates": (-3.5, 0, 3.5),
    })

    benchmark = DummyBenchmark()
    benchmark.test(planner)

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # from spider.planner_zoo.LatticePlanner import LatticePlanner
    # from spider.elements.map import RoutedLocalMap, Lane
    # from spider.elements.Box import TrackingBoxList, TrackingBox
    # from spider.elements.vehicle import VehicleState, Transform, Location, Rotation, Vector3D
    # import spider.visualize as vis
    # #################### 输入信息的初始化 ####################
    # # 定位信息
    # ego_veh_state = VehicleState(
    #     transform=Transform(
    #         location=Location(5., 1., 0),
    #         # location=Location(5., 0.01, 0),
    #         rotation=Rotation(0, 0, 0)
    #     ),
    #     velocity=Vector3D(0, 0, 0),
    #     acceleration=Vector3D(0, 0, 0)
    # )
    # # 地图信息
    # local_map = RoutedLocalMap()
    # for idx, yy in enumerate([-3.5, 0, 3.5]):
    #     xs = np.arange(0, 300.1, 1.0)
    #     cline = np.column_stack((xs, np.ones_like(xs) * yy))
    #     lane = Lane(idx, cline, width=3.5, speed_limit=60 / 3.6)
    #     local_map.lanes.append(lane)
    # # 感知信息
    # obstacles = TrackingBoxList()
    # obstacles.append(TrackingBox(obb=(50, 0, 5, 2, np.arctan2(0.2, 5)), vx=5, vy=0.2))
    # obstacles.append(TrackingBox(obb=(100, 0, 5, 2, 0), vx=5, vy=0))
    # obstacles.append(TrackingBox(obb=(200, -10, 1, 1, np.pi/2), vx=0, vy=1.0)) # 横穿马路
    #
    # lattice_planner = LatticePlanner()
    # lattice_planner.configure({
    #     "steps":15,
    #     "dt":0.2,
    #     "end_s_candidates": (20, 40, 60),
    #     "end_l_candidates": (-3.5, 0, 3.5),
    # })
    # lattice_planner.set_local_map(local_map)
    #
    # plt.figure(figsize=(14, 4))
    # plt.axis('equal')
    # plt.tight_layout()
    # snapshot = vis.SnapShot(True, 15)
    # ################## main loop ########################
    # while True:
    #     if ego_veh_state.x() > 250: break
    #
    #     # 地图信息更新
    #
    #     # 感知信息更新，这里假设完美感知+其他车全部静止
    #
    #     # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
    #     # ego_veh_state = ...
    #
    #     traj = lattice_planner.plan(ego_veh_state, obstacles)  # , local_map)
    #
    #     # 可视化
    #     plt.cla()
    #     for lane in local_map.lanes:
    #         plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1.5)  # 画地图
    #     # vis.draw_ego_vehicle(ego_veh_state, color='green', fill=True, alpha=0.2, linestyle='-', linewidth=1.5) # 画自车
    #
    #     for tb in obstacles:
    #         vis.draw_boundingbox(tb, color='black',fill=True, alpha=0.1, linestyle='-', linewidth=1.5)# 画他车
    #         # 画他车预测轨迹
    #         tb_pred_traj = np.column_stack((tb.x + traj.t * tb.vx, tb.y+traj.t*tb.vy))
    #         vis.draw_polyline(tb_pred_traj,show_buffer=True,buffer_dist=tb.width*0.5,buffer_alpha=0.1, color='C3')
    #
    #     vis.draw_ego_history(ego_veh_state,'-', lw=1, color='gray')# 画自车历史
    #     vis.draw_trajectory(traj,'.-', show_footprint=True, color='C2')# 画轨迹
    #     vis.draw_ego_vehicle(ego_veh_state, color='C0', fill=True, alpha=0.3, linestyle='-', linewidth=1.5) # 画自车
    #     # plt.axis('equal')
    #     # plt.tight_layout()
    #     vis.ego_centric_view(ego_veh_state.x(), ego_veh_state.y(), [-20,80], [-5,5])
    #     # plt.xlim([ego_veh_state.x() - 20, ego_veh_state.x() + 80])
    #     # plt.ylim([ego_veh_state.y() - 5, ego_veh_state.y() + 5])
    #     plt.pause(0.01)
    #     snapshot.snap(plt.gca())
    #
    #     # 控制+定位，假设完美控制到下一个轨迹点
    #     ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
    #         = traj.x[1], traj.y[1], traj.heading[1]
    #     ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
    #         = traj.v[1], traj.a[1], traj.curvature[1]
    #
    #     for tb in obstacles:
    #         tb.set_obb([tb.x+tb.vx*traj.dt, tb.y+tb.vy*traj.dt, tb.length, tb.width, tb.box_heading])
    #
    # plt.close()
    # snapshot.print(3,2,figsize=(15,6))
    # plt.show()



if __name__ == '__main__':
    demo()

