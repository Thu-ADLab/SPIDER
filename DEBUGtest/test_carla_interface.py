
from spider.interface.carla.CarlaInterface import CarlaInterface
import carla

from tqdm import tqdm

# host = '192.168.3.107'
host = "101.5.228.94"
cport = 2000
tmport = 8000
env = CarlaInterface(host, cport, tmport)

# env.spawn_hero()
# env.bev_spectator(250)
maps = env.client.get_available_maps()
print(maps)

env.load_map('Town10HD_Opt', map_layers=carla.MapLayer.Ground)
# env.random_weather(True)

env.spawn_hero(autopilot=True)
# env.bev_spectator(20, 5, 5)

for i in tqdm(range(1000)):

    env.world.tick()

    env.side_view_spectator()
    # env.third_person_spectator()
    # env.first_person_spectator()
    # env.bev_spectator()

    # loc = env.hero.get_location()
    # loc.z = 20
    # env.spectator.set_location(loc)

env.destroy()
pass