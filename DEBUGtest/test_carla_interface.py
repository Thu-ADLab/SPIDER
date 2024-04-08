import matplotlib.pyplot as plt
import random
import carla
import pygame
from tqdm import tqdm

import spider.planner_zoo
from spider.interface.carla.CarlaInterface import CarlaInterface
from spider.planner_zoo import LatticePlanner
import spider.visualize as vis

import pygame


# host = '192.168.3.107' # 无线局域网
host = '192.168.189.13' # 有线局域网
# host = "101.5.228.94"
# host = '127.0.0.1'
cport = 2000
tmport = 8000
# random.seed(123)
# random.seed(40)
random.seed(11)
env = CarlaInterface(
    host,
    cport,
    tmport,
    recording=True
)

planner = LatticePlanner({
    "steps": 20,
    "dt": 0.2,
    "ego_veh_length": env.ego_size[0], # ！！！这边还没有spawn hero
    "ego_veh_width": env.ego_size[1],
    "end_s_candidates": (10, 30),#(20,40,60),
    "end_l_candidates": (0,),#(-0.5,0,0.5),
    "end_v_candidates": tuple(i*60/3.6/3 for i in range(4)),
    "constraint_flags": {},
    "print_info": False
})

SPIDER_PLOT = 0
# planner = spider.planner_zoo.DummyPlanner()

try:
# if 1:

    maps = env.client.get_available_maps()
    print(maps)

    map_name = 'Town10HD'
    env.load_map(map_name)

    # print(env.map.name)
    # if env.map is not None and not (map_name in env.map.name):
    #     print("loading map...")
    #     # env.load_map('Town10HD_Opt', map_layers=carla.MapLayer.Ground)
    #     env.load_map(map_name)
    # else:
    #     env.destroy()
    # env.random_weather(True)

    env.spawn_hero(autopilot=False)
    env.generate_traffic(50,5)

    env.bev_spectator(30, 5, 5, True)
    # env.side_view_spectator(left=False)
    # env.third_person_spectator()
    # env.first_person_spectator()

    env.viewer.change_view("first_person")
    vis.figure()
    # display = None
    display = pygame.display.set_mode(
        env.viewer.image_size,
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    snapshot = vis.SnapShot(False, record_video=True, video_path='carla_visualize_recording.avi', fps=20)


    ##############
    # test:不允许自动换道
    for actor in env.npc_vehicles:
        env.traffic_manager.auto_lane_change(actor, False)
    ###########

    delta_steps = 1  # int(planner.dt /0.05)#/ 0.05)
    traj = None
    for i in tqdm(range(1000)):

        env.world.tick()


        env.render(display)
        pygame.display.flip()


        ego_veh_state, tb_list, routed_local_map = env.wrap_observation()
        if traj is None or i%delta_steps==0:
            traj = planner.plan(ego_veh_state, tb_list, routed_local_map)
        else:
            traj.x = traj.x[1:]
            traj.y = traj.y[1:]
            traj.heading = traj.heading[1:]
            traj.v = traj.v[1:]

        # assert traj is not None

        if traj is not None:
            for x, y in zip(traj.x, traj.y):
                loc = carla.Location(x=x, y=y, z=env.hero.get_location().z + 0.5)
                env.world.debug.draw_point(loc, size=0.1, life_time=0.1)

        env.conduct_trajectory(traj, ego_veh_state)


        if (traj is not None) and SPIDER_PLOT:
            # pass
            plt.cla()
            vis.draw_ego_vehicle(ego_veh_state, color='C0', fill=True, alpha=0.3, linestyle='-', linewidth=1.5)
            vis.draw_trackingbox_list(tb_list,draw_prediction=False)
            vis.draw_local_map(routed_local_map)
            vis.draw_trajectory(traj, '.-', show_footprint=True, color='C2')  # 画轨迹
            vis.ego_centric_view(ego_veh_state.x(), ego_veh_state.y(),(-50,50),(-50,50))
            plt.pause(0.01)
            snapshot.snap(plt.gca())

        if env.has_arrived():
            print("Hero has arrived!")
            break


except Exception as e:
    print(e)

finally:
    env.destroy()

    if "snapshot" in dir():
        snapshot.release_writer()


