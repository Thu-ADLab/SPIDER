# from spider.interface.BaseBenchmark import BaseBenchmark
from typing import Sequence

import random
import sys
import warnings

import carla

from spider.interface.carla.common import *


class CarlaInterface:

    _hero_name = "hero"
    _manual_npc_name = "npc"
    _auto_npc_name = "autopilot"
    _walker_name = "walker"

    _target_options = ["npc", "hero", "all"]

    def __init__(self,
                 client_host='127.0.0.1',
                 client_port=2000,
                 traffic_manager_port=8000,
                 rendering=True) -> None:

        self.client_host = client_host
        self.client_port = client_port
        self.traffic_manager_port = traffic_manager_port

        self.client = carla.Client(client_host, client_port)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(traffic_manager_port)

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self.hero = None

        self._sync = True  # 同步模式
        self._synchronous_master = True # 不懂什么意思

        self._rendering = rendering

        settings = self.world.get_settings()
        settings.no_rendering_mode = not self._rendering
        self.world.apply_settings(settings)
        # self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)


        self.viewer = None # the main viewer for visualization

        self._default_vehicle_bp_filter = "vehicle.*"
        self._default_walker_bp_filter = "walker.pedestrian.*"

    @property
    def actors(self):
        return self.world.get_actors()

    @property
    def vehicles(self):
        return self.actors.filter('vehicle.*')

    @property
    def npc_vehicles(self) -> Sequence[carla.Vehicle]:
        # 其实也可以是记录下来spawn时候的actor id，然后用id来索引
        all_veh = self.vehicles
        return [veh for veh in all_veh if veh.attributes['role_name'] != self._hero_name]

    @property
    def blueprint_library(self):
        return self.world.get_blueprint_library()

    def get_blueprints_by_type(self, filter_string, default_filter=None):
        if filter_string is None:
            bps = []
        else:
            bps = self.blueprint_library.filter(filter_string)

        if len(bps) == 0:
            warnings.warn("No blueprint meets the given filter. Use default filter instead.")
            if default_filter is None:
                bps = self.blueprint_library
            else:
                bps = self.blueprint_library.filter(default_filter)
        return bps

    @property
    def spectator(self):
        return self.world.get_spectator()

    # todo: 把以下四个视角的spectator切换都转为一个函数
    def bev_spectator(self, height=50, lon_offset=0, lat_offset=0):
        if self.hero is None:
            hero_transform = self.spectator.get_transform()
        else:
            hero_transform = self.hero.get_transform()

        spec_transform = bev_transform(height,lon_offset,lat_offset, absolute=True, hero_transform=hero_transform)
        self.spectator.set_transform(spec_transform)

    def third_person_spectator(self, height=2.5, back_distance=5.5, lat_offset=0., pitch=-8.0,):
        if self.hero is None:
            hero_transform = self.spectator.get_transform()
        else:
            hero_transform = self.hero.get_transform()
        spec_transform = third_person_view_transform(height, back_distance, lat_offset,pitch,
                                                     absolute=True, hero_transform=hero_transform)
        self.spectator.set_transform(spec_transform)

    def first_person_spectator(self, height=1.5, lon_offset=2.5, lat_offset=0., pitch=-8):
        if self.hero is None:
            hero_transform = self.spectator.get_transform()
        else:
            hero_transform = self.hero.get_transform()
        spec_transform = first_person_view_transform(height, lon_offset, lat_offset, pitch,
                                                     absolute=True, hero_transform=hero_transform)
        self.spectator.set_transform(spec_transform)

    def side_view_spectator(self,lon_offset=5, lat_offset=12,left=False,):
        if self.hero is None:
            hero_transform = self.spectator.get_transform()
        else:
            hero_transform = self.hero.get_transform()
        spec_transform = side_view_transform(lon_offset, lat_offset, left,
                                            absolute=True, hero_transform=hero_transform)
        self.spectator.set_transform(spec_transform)

    def get_actors_by_type(self, filter_string):
        # todo: 完成
        pass

    def get_actors_by_ids(self, ids):
        return self.world.get_actors(ids)

    @property
    def weather(self): return self.world.get_weather()

    def random_weather(self, preset_combination=True):
        weather = generate_random_weather(preset_combination)
        self.world.set_weather(weather)

    def set_weather(self, sun:Union[str, Sequence, None]=None, weather:Union[str, Sequence, None]=None):
        new_weather = modify_weather_param(self.world.get_weather(), sun, weather)
        self.world.set_weather(new_weather)

    @property
    def available_maps(self):
        return self.client.get_available_maps() # 分层地图用后缀_Opt标识

    def load_map(self, map_name, reset_settings=True, map_layers=carla.MapLayer.All):
        try:
            self.world = self.client.load_world(map_name, reset_settings, map_layers) # little bug: cannot use | because the C++ signature requires enum
        except RuntimeError:
            warnings.warn("Could not load map.\nPlease ensure that the map name is one of {}".format(self.available_maps))
        self.destroy()
        # if layer_flag is None:
        #     self.map = self.world.get_map(map_name)
        # else:
        #     self.map = self.world.get_map(map_name, layer_flag)

        # todo: 完成，设置town05之类
        pass

    def attach_all_view_cameras(self):
        # 重新取个名字。总共6个视角的相机，用来做BEV视角的
        pass

    def attach_camera_to_hero(self, view): #camera_bp, pos=carla.Location(x=1.5, z=2.4)):
        pass
        # if self.hero is None:
        #     warnings.warn("Hero vehicle is not spawned yet.")
        #     return None
        #
        # transform = carla.Transform(pos)
        # transform.rotation = transform.get_forward_vector().rotate(carla.Rotation(0.0, 180.0, 0.0))
        # cam = self.world.spawn_actor(camera_bp, transform, attach_to=self.hero)
        # self.sensors.append(cam)
        # return cam

    def attach_collision_sensor_to_hero(self):
        pass

    def set_autopilot(self,  target='all', enable=True):
        if not (target in self._target_options):
            raise ValueError("target must be one of {}".format(self._target_options))
        if target == 'hero':
            if self.hero is not None:
                self.hero.set_autopilot(enable)
            else:
                warnings.warn("Hero vehicle is not spawned yet. Can not set autopilot mode!")
        elif target == 'npc':
            for veh in self.npc_vehicles:
                veh.set_autopilot(enable)
        else: # all
            for veh in self.vehicles:
                veh.set_autopilot(enable)

    def set_autolight(self):
        pass


    def spawn_hero(self, ego_x=None, ego_y=None, ego_yaw=None, blueprint_filter="vehicle*",
                   autopilot=False, autolight=True, only_four_wheel=True):

        if ego_x is None or ego_y is None or ego_yaw is None:
            # did not provide ego position, randomly pick one
            print("Provide no transform of ego vehicle, try to randomly pick one.")
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        else:
            initial_transform = carla.Transform()
            initial_transform.location = carla.Location(ego_x, ego_y, 2.5)
            # todo: how to set the location height?
            initial_transform.rotation = carla.Rotation(pitch=0.0, yaw=ego_yaw, roll=0.0)
            spawn_point = initial_transform

        if self.hero is not None:
            print("Found existing hero. Destroying...")
            self.destroy_hero()

        # Get a random blueprint.
        blueprints = self.get_blueprints_by_type(blueprint_filter, self._default_vehicle_bp_filter)
        # blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3') # dedault vehicle.tesla.model3
        if only_four_wheel:
            blueprints = filter_four_wheel_vehicles(blueprints)
        blueprint = random.choice(blueprints)

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', self._hero_name)

        self.hero = self.world.try_spawn_actor(blueprint, spawn_point)  # 有可能存在无法spawn的可能
        modify_vehicle_physics(self.hero)

        self.hero.set_autopilot(autopilot)
        if autolight:
            set_autolight(self.hero, self.traffic_manager)

        if self.hero is None:
            raise RuntimeError(
                "Player is not spawn. It might be due to incorrect position input or existing space occupancy.")

        if self._sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def spawn_npc_vehicles(self, vehicle_num, vehicle_filter="vehicle*", spawn_points=None,
                           autopilot=True, autolight=True, only_four_wheel=False):
        if vehicle_num <= 0:
            return []

        if spawn_points is None:
            spawn_points = self.map.get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if vehicle_num < number_of_spawn_points:
                random.shuffle(spawn_points)
            elif vehicle_num > number_of_spawn_points:
                warnings.warn('requested %d vehicles, but could only find %d spawn points' % (vehicle_num,number_of_spawn_points))
        else:
            assert len(spawn_points) >= vehicle_num

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        blueprints = self.get_blueprints_by_type(vehicle_filter, self._default_vehicle_bp_filter)
        if only_four_wheel:
            blueprints = filter_four_wheel_vehicles(blueprints)
        for n, transform in enumerate(spawn_points):
            if n >= vehicle_num:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            if autopilot:
                blueprint.set_attribute('role_name', self._auto_npc_name)
            else:
                blueprint.set_attribute('role_name', self._manual_npc_name)

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, autopilot, self.traffic_manager_port)))

        batch_responses = self.client.apply_batch_sync(batch, self._synchronous_master) # 执行指令
        vehicles_id_list = []
        error_num = 0
        for response in batch_responses:
            if response.error:
                error_num += 1
            else:
                vehicles_id_list.append(response.actor_id)

        if error_num > 0:
            warnings.warn("%d vehicles are not correctly spawned." % error_num)

        # car light
        if autolight:
            all_vehicle_actors = self.world.get_actors(vehicles_id_list)
            for actor in all_vehicle_actors:
                # self.traffic_manager.update_vehicle_lights(actor, True)
                set_autolight(actor, self.traffic_manager, True)

        # 下面要加吗？
        if self._sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        return vehicles_id_list

    def spawn_npc_walkers(self, walker_num, walker_filter='walker.pedestrian.*', spawn_points=None,
                          running_rate=0.0, crossing_rate=0.0, invincible=False):
        if walker_num <= 0:
            return []

        if spawn_points is None:

            spawn_points = []
            for i in range(walker_num):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
        else:
            assert len(spawn_points) >= walker_num

        blueprints = self.get_blueprints_by_type(walker_filter, self._default_walker_bp_filter)
        percentagePedestriansRunning = running_rate  # how many pedestrians will run
        percentagePedestriansCrossing = crossing_rate  # how many pedestrians will walk through the road

        batch = []
        walker_speed = []

        SpawnActor = carla.command.SpawnActor
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)

            walker_bp.set_attribute('role_name', self._walker_name)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'true' if invincible else 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)

        walkers_id_list = []
        walker_speed2 = []
        error_num = 0
        for i in range(len(results)):
            if results[i].error:
                error_num += 1
            else:
                walkers_id_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2


        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        for i in range(len(walkers_id_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_id_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                error_num += 1
            else:
                walkers_id_list[i]["con"] = results[i].actor_id

        if error_num > 0:
            warnings.warn("%d vehicles are not correctly spawned." % error_num)
        # 4. we put together the walkers and controllers id to get the objects from their id
        all_id = []
        for i in range(len(walkers_id_list)):
            all_id.append(walkers_id_list[i]["con"])
            all_id.append(walkers_id_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if self._sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))


    def generate_traffic(self, vehicle_num=30, walker_num=10):
        # todo: add more settings
        self.spawn_npc_vehicles(vehicle_num,)
        self.spawn_npc_walkers(walker_num)

    def remove_all_npc(self):
        npc_names = [self._manual_npc_name, self._auto_npc_name, self._walker_name]
        for actor in self.actors:
            if actor.attributes.get('role_name', None) in npc_names:
                # actor.attributes is a dict, get() uses None as a default value
                actor.destroy()
        print("Removed all NPCs.")

    def destroy_hero(self):
        # todo:删除player和所有与player连接的传感器actor
        if self.hero is not None:
            self.hero.destroy()
        self.hero = None
        print("Destroyed hero.")

    def destroy(self):
        self.destroy_hero()
        self.remove_all_npc()


