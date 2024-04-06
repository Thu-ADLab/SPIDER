# from spider.interface.BaseBenchmark import BaseBenchmark
import math
from typing import Sequence, Tuple

import random
import sys
import warnings

import carla

import spider.elements.box
from spider.interface.carla.common import *
from spider.interface.carla.visualize import Viewer
from spider.elements import RoutedLocalMap, VehicleState, TrackingBoxList, TrackingBox, Lane, Trajectory
from spider.utils.geometry import resample_polyline
# from spider.interface.carla._route_utils import GlobalRoutePlanner


class CarlaInterface:

    _hero_name = "hero"
    _manual_npc_name = "npc"
    _auto_npc_name = "autopilot"
    _walker_name = "walker"

    _target_options = ["npc", "hero", "all"]

    _map_resolution = 1.0

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
        settings.synchronous_mode = self._sync  # Enables synchronous mode
        if self._sync:
            settings.fixed_delta_seconds = 0.05  # 可变时间步长
        self.world.apply_settings(settings)

        self.traffic_manager.set_synchronous_mode(self._sync)
        # self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        self.viewer = None # the main viewer for visualization


        # navigation settings
        self.origin: carla.Location = None
        self.destination: carla.Location = None
        self.route: Sequence[Tuple[carla.Waypoint, RoadOption]] = None
        self._route_arr = None
        self._router = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=self._map_resolution)

        # control settings
        self._control_dt = 0.2 # 控制时间间隔
        self._controller:VehiclePIDController = None # VehiclePIDController(self._vehicle)


        self._default_vehicle_bp_filter = "vehicle.*"
        self._default_walker_bp_filter = "walker.pedestrian.*"

    ############## useful functions ###############

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
    def ego_size(self):
        '''
        Return the size of the ego vehicle.
        '''
        if self.hero is None:
            warnings.warn("Hero vehicle is not spawned yet.")
            return (5.0, 2.0)
        length = self.hero.bounding_box.extent.x * 2.0
        width = self.hero.bounding_box.extent.y * 2.0
        return (length, width)


    def get_random_point(self) -> carla.Transform:
        spawn_points = self.map.get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        return spawn_point

    def get_nearest_waypoint(self) -> carla.Waypoint:
        assert self.hero is not None, "Hero vehicle is not spawned yet."
        current_location = self.hero.get_location()  # general
        return self.world.get_map().get_waypoint(current_location)


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
    def bev_spectator(self, height=50, lon_offset=0, lat_offset=0, vertical=True):
        if self.hero is None:
            hero_transform = self.spectator.get_transform()
        else:
            hero_transform = self.hero.get_transform()

        spec_transform = bev_transform(height,lon_offset,lat_offset,vertical=vertical, absolute=True, hero_transform=hero_transform)
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
        return self.world.get_actors().filter(filter_string)

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

    @property
    def debug(self):
        return self.world.debug

    def attach_all_view_cameras(self):
        # 重新取个名字。总共6个视角的相机，用来做BEV视角的
        pass

    def attach_camera_to_hero(self, view): #camera_bp, pos=carla.Location(x=1.5, z=2.4)):
        pass

    def attach_collision_sensor_to_hero(self):
        pass

    def set_autopilot(self, target:Union[str, carla.Actor]='all', enable=True):
        if isinstance(target, str):
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
        else:
            target.set_autopilot(enable)

    def set_autolight(self, target:Union[str, carla.Actor]='all', enable=True):
        '''
        Notice that only those vehicles that registered in the traffic manager can be set autolight.
        '''

        if isinstance(target, str):
            if not (target in self._target_options):
                raise ValueError("target must be one of {}".format(self._target_options))
            if target == 'hero':
                if self.hero is not None:
                    set_autolight(self.hero, self.traffic_manager, enable)
                else:
                    warnings.warn("Hero vehicle is not spawned yet. Can not set autolight mode!")
            elif target == 'npc':
                for veh in self.npc_vehicles:
                    set_autolight(veh, self.traffic_manager, enable)
            else: # all
                for veh in self.vehicles:
                    set_autolight(veh, self.traffic_manager, enable)
        else:
            set_autolight(target, self.traffic_manager, enable)


    def spawn_viewer(self, viewed_object=None, sensor_type="camera_rgb", view="third_person",
                 recording=False, image_size=(1280, 720), lidar_range=80):
        if viewed_object is None:
            if self.hero is not None:
                viewed_object = self.hero
            else:
                warnings.warn("No object to view. Please provide a viewed object first.")
                return
        if self.viewer is not None:
            print("Found existing viewer. Destroying...")
            self.viewer.destroy()
        self.viewer = Viewer(viewed_object, sensor_type, view, recording, image_size, lidar_range)

    def spawn_hero(self, ego_x=None, ego_y=None, ego_yaw=None, blueprint_filter="vehicle*",
                   destination:carla.Location=None, autopilot=False, autolight=True, only_four_wheel=True):

        if ego_x is None or ego_y is None or ego_yaw is None:
            # did not provide ego position, randomly pick one
            print("Provide no transform of ego vehicle, try to randomly pick one.")
            spawn_point = self.get_random_point()
            spawn_point.location.z += 0.5
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

        hero = self.world.try_spawn_actor(blueprint, spawn_point)  # 有可能存在无法spawn的可能
        if hero is None:
            raise RuntimeError(
                "Player is not spawn. It might be due to incorrect position input or existing space occupancy.")

        self.set_hero(hero, destination, autopilot, autolight)

        # modify_vehicle_physics(self.hero)
        # self.hero.set_autopilot(autopilot)
        # if autolight:
        #     set_autolight(self.hero, self.traffic_manager)
        #
        #
        # # routing
        # self.origin = spawn_point.location
        # if not autopilot:
        #     route_length = 0
        #     while route_length < 100 / self._map_resolution:
        #         self.destination = self.get_random_point().location if destination is None else destination
        #         self.route = self._router.trace_route(self.origin, self.destination) # waypoint, road_option
        #         route_length = len(self.route)
        #     self._route_arr = waypointseq2array([wp_info[0] for wp_info in self.route])
        #     self._route_arr = resample_polyline(self._route_arr, self._map_resolution)
        #     # self.route = self.map.compute_route(self.origin.location, self.destination.location)
        #
        # # control
        # self._controller = VehiclePIDController(self.hero, self._control_dt)
        #
        # # set the main_viewer
        # self.spawn_viewer(self.hero)
        # # self.spawn_viewer(self.hero, sensor_type="camera_rgb", view="bird_eye")
        # # self.spawn_viewer(self.hero, sensor_type="camera_rgb", view="right_side")
        # # self.spawn_viewer(self.hero, sensor_type="lidar",view="bird_eye", image_size=(640,360))
        # # self.spawn_viewer(self.hero, sensor_type="camera_log_gray_depth", view="first_person",
        # #                   image_size=(640, 360), recording=True)

        if self._sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def set_hero(self, hero: carla.Actor, destination:carla.Location=None, autopilot=False, autolight=True):
        if self.hero is not None:
            print("Found existing hero. Destroying...")
            self.destroy_hero()

        self.hero = hero
        modify_vehicle_physics(self.hero)
        self.hero.set_autopilot(autopilot)
        if autolight:
            set_autolight(self.hero, self.traffic_manager)

        # routing
        self.origin = hero.get_location()
        if not autopilot:
            route_length = 0
            while route_length < 100 / self._map_resolution:
                self.destination = self.get_random_point().location if destination is None else destination
                self.route = self._router.trace_route(self.origin, self.destination)  # waypoint, road_option
                route_length = len(self.route)
            self._route_arr = waypointseq2array([wp_info[0] for wp_info in self.route])
            self._route_arr = resample_polyline(self._route_arr, self._map_resolution)
            # self.route = self.map.compute_route(self.origin.location, self.destination.location)
        else:
            print("autopilot actor does not have a destination")

        # control
        self._controller = VehiclePIDController(self.hero, self._control_dt)

        # set the main_viewer
        self.spawn_viewer(self.hero)

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
        vehicles = self.world.get_actors().filter("vehicle.*")
        walkers = self.world.get_actors().filter("walker.*")
        # for walker in walkers:
        #     walker.stop() # stop the navigation of walkers

        hero_id = self.hero.id if self.hero is not None else None
        all_ids = [veh.id for veh in vehicles if veh.id != hero_id]
        all_ids += [walker.id for walker in walkers]

        # self.client.apply_batch([carla.command.DestroyActor(x) for x in all_ids])
        for actor in self.world.get_actors(all_ids):
            actor.destroy()

        print("Removed all NPCs.")

    def reset(self):
        # todo: 写一个config，储存这些信息。
        self.destroy()
        self.spawn_hero()
        self.generate_traffic()




    def render(self, display=None):
        if self._rendering:
            # main viewer rendering....
            if self.viewer is None:
                print("No viewer spawned yet, can not render")
            else:
                display = self.viewer.render(display)
            # other sensors rendering, to complete...
        return display

    def destroy_hero(self):
        # todo:删除player和所有与player连接的传感器actor
        if self.viewer is not None:
            self.viewer.destroy()
        if self.hero is not None:
            self.hero.destroy()
        self.hero = None
        self.viewer = None
        print("Destroyed hero.")

    def destroy(self):
        self.destroy_hero()
        self.remove_all_npc()

    ############## spider interface ###############

    # def get

    def wrap_observation(self, valid_distance=100) \
            -> Tuple[VehicleState, Union[TrackingBoxList, "OccupancyGrid2D"], RoutedLocalMap]:

        assert self.hero is not None, "Hero not spawned!"

        # ego vehicle
        info = get_actor_info(self.hero)
        ego_veh_state = VehicleState.from_kine_states(
            info["x"], info["y"], info["yaw"], info["vx"], info["vy"],# info["ax"], info["ay"],
            length=info["length"], width=info["width"] # 注意，这里获取的length和width，在外部也要用这个值
        )
        egox, egoy = info["x"], info["y"]

        # other vehicle
        roi = (egox - valid_distance, egoy - valid_distance,
               egox + valid_distance, egoy + valid_distance)
        vehicles = self.world.get_actors().filter('vehicle.*')
        walkers = self.world.get_actors().filter("walker.*")
        hero_id = self.hero.id
        all_npc_ids = [veh.id for veh in vehicles if veh.id != hero_id]
        all_npc_ids += [walker.id for walker in walkers]

        tb_list = TrackingBoxList()
        for actor in self.world.get_actors(all_npc_ids):
            info = get_actor_info(actor)
            if (roi[0] < info["x"] < roi[2]) and (roi[1] < info["y"] < roi[3]):
                tb_list.append(TrackingBox(obb=(info["x"], info["y"], info["length"], info["width"], info["yaw"]),
                                           vx=info["vx"], vy=info["vy"], id=actor.id))

        routed_local_map = RoutedLocalMap()
        if self._route_arr is not None and self.route is not None:
            routed_local_map.route = self.route
            routed_local_map.route_arr = self._route_arr
            route_virtual_lane = Lane(-1, routed_local_map.truncate_route_arr(egox,egoy))
            routed_local_map.lanes.append(route_virtual_lane)

        # todo: 目前没有补充，lanes仅包含routing给定的。但其实应该有其他车道。找到距离最近的route里面的waypoint，找相邻车道
        # nearest_wp = self.get_nearest_waypoint()
        # neighboring_wps = get_neighboring_waypoints(nearest_wp)
        # for idx, wp in enumerate(neighboring_wps):
        #     in_node, out_node = self._router._localize(wp.transform.location)
        #     edge = self._router._graph.edges[in_node, out_node]
        #     path = edge['path']#[edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
        #     # path is a list of waypoints
        #     routed_local_map.lanes.append(Lane(idx, waypointseq2array(path)))

        return ego_veh_state, tb_list, routed_local_map

    def convert_to_action(self, trajectory:Trajectory):
        assert self.hero is not None, "Hero not spawned!"

        if trajectory is None:
            print("\033[31m Provide no trajectory to track! Try to hold still... \033[0m")
            return self._controller.get_fallback_control()


        # 这里搞成插值最好，目前是找最近索引，相当于是最近邻插值
        # target_index = int(round(trajectory.dt / self._control_dt))

        hero_trans = self.hero.get_transform()
        idx = 1
        yaw_deg = trajectory.heading[idx] * 180 / math.pi
        target_point_transform = carla.Transform(
            carla.Location(x=trajectory.x[idx], y=trajectory.y[idx], z=hero_trans.location.z),
            carla.Rotation(roll=hero_trans.rotation.roll, yaw=yaw_deg, pitch=hero_trans.rotation.pitch)
        )
        target_speed = trajectory.v[1] * 3.6 # qzl: in km/h

        control = self._controller.run_step(target_speed, target_point_transform)
        control.manual_gear_shift = False
        return control

    def conduct_trajectory(self, trajectory:Trajectory):
        control = self.convert_to_action(trajectory)
        self.hero.apply_control(control)
        # print(control)

    def has_arrived(self, dist_thresh=5) -> bool:
        # 这里只是粗略位置判断
        ego_x, ego_y = self.hero.get_location().x, self.hero.get_location().y
        des_x, des_y = self.destination.x, self.destination.y
        if (des_x - ego_x) ** 2 + (des_y - ego_y) ** 2 < dist_thresh**2:
            return True
        else:
            return False
    # return location.x, location.y,  length, width, rotation.yaw, velocity.x, velocity.y

