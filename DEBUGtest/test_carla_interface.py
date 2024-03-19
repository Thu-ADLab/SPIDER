
from spider.interface.carla.CarlaInterface import CarlaInterface
import carla
import pygame

from tqdm import tqdm

# host = '192.168.3.107' # 无线局域网
host = '192.168.189.13' # 有线局域网
# host = "101.5.228.94"
# host = '127.0.0.1'
cport = 2000
tmport = 8000
env = CarlaInterface(host, cport, tmport)

# if 1:
try:

    # env.spawn_hero()
    # env.bev_spectator(250)
    maps = env.client.get_available_maps()
    print(maps)

    # map_name = 'Town10HD'
    # print(env.map.name)
    # if env.map is not None and not (map_name in env.map.name):
    #     print("loading map...")
    #     # env.load_map('Town10HD_Opt', map_layers=carla.MapLayer.Ground)
    #     env.load_map(map_name)

    # env.random_weather(True)

    env.spawn_hero(autopilot=True)
    env.generate_traffic(50)
    env.bev_spectator(30, 5, 5, True)
    env.viewer.change_view("bird_eye")

    for i in tqdm(range(1000)):

        env.world.tick()

        loc1 = env.hero.get_transform().location
        loc2 = carla.Location(x=loc1.x+10, y=loc1.y+10, z=loc1.z)
        current_w = env.map.get_waypoint(env.hero.get_location())
        next_w = current_w.next(20.0)[0]

        # next_w = env.map.get_waypoint(env.hero.get_location(),
        #                               lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk)

        env.world.debug.draw_line(env.hero.get_location(),next_w.transform.location,0.1,life_time=0.1)

        env.render()
        pygame.display.flip()

        # env.side_view_spectator(left=False)
        # env.third_person_spectator()
        # env.first_person_spectator()
        # env.bev_spectator()

        # loc = env.hero.get_location()
        # loc.z = 20
        # env.spectator.set_location(loc)

finally:
    env.destroy()

