import random
from typing import Union,Sequence

import carla
import numpy as np
import math

from spider.interface.carla.presets import *


# from _weather_utils import *

################### blueprint #####################
def is_uncommon_vehicle(bp):
    return int(bp.get_attribute('number_of_wheels')) != 4 or bp.id.endswith('isetta') or bp.id.endswith('carlacola') \
        or bp.id.endswith('cybertruck') or bp.id.endswith('t2')

def filter_four_wheel_vehicles(blueprints):
    return [bp for bp in blueprints if bp.id.startswith('vehicle') and int(bp.get_attribute('number_of_wheels')) == 4]

################## transform for views #########################
def first_person_view_transform(height=1.5, lon_offset=2.5, lat_offset=0., pitch=-8,
                                absolute=False, hero_transform=None):
    if not absolute:
        return carla.Transform(carla.Location(x=lon_offset, y=lat_offset, z=height),
                               carla.Rotation(pitch=pitch))
    else:
        assert hero_transform is not None, "hero_transform must be provided for absolute third person view transform"
        yaw = hero_transform.rotation.yaw
        yaw_rad = np.radians(yaw)
        x = hero_transform.location.x + lon_offset * np.cos(yaw_rad) + lat_offset * np.sin(yaw_rad)
        y = hero_transform.location.y + lon_offset * np.sin(yaw_rad) - lat_offset * np.cos(yaw_rad)

        return carla.Transform(carla.Location(x=x, y=y, z=height),
                               carla.Rotation(pitch=pitch, yaw=yaw, roll=0))


def third_person_view_transform(height=2.8, back_distance=5.5, lat_offset=0., pitch=-10.0,
                                absolute=False, hero_transform=None):
    # 对于lat offset 车左为正方向。对于lon offset 车前为正方向。
    # 若absolute为True，则返回的transform为世界坐标系，否则为hero的自车坐标系下的相对transform。
    back_distance = np.abs(back_distance)
    if not absolute:
        return carla.Transform(carla.Location(x=-back_distance, y=lat_offset, z=height),
                               carla.Rotation(pitch=pitch))
    else:
        assert hero_transform is not None, "hero_transform must be provided for absolute third person view transform"
        yaw = hero_transform.rotation.yaw
        yaw_rad = np.radians(yaw)
        x = hero_transform.location.x - back_distance * np.cos(yaw_rad) + lat_offset * np.sin(yaw_rad)
        y = hero_transform.location.y - lat_offset * np.cos(yaw_rad) - back_distance * np.sin(yaw_rad)
        return carla.Transform(carla.Location(x=x, y=y, z=height),
                               carla.Rotation(pitch=pitch, yaw=yaw, roll=0))


def side_view_transform(lon_offset=5., lat_offset=10.,left=False, absolute=False, hero_transform=None):
    # 生成侧边视角的相机transform
    # left为True表示在车辆左侧，left为False表示在车辆右侧

    lat_offset = np.abs(lat_offset)
    if not absolute:
        if left:
            return carla.Transform(carla.Location(x=lon_offset, y=-lat_offset, z=1.0),
                                   carla.Rotation(yaw=90))
        else:
            return carla.Transform(carla.Location(x=lon_offset, y=+lat_offset, z=1.0),
                                   carla.Rotation(yaw=-90))

    else:
        assert hero_transform is not None, "hero_transform must be provided for absolute side view transform"

        # 获取自车的朝向（yaw角度）
        yaw = hero_transform.rotation.yaw
        yaw_rad = np.radians(yaw)

        # 计算侧边视角相机的位置偏移
        lat_offset = lat_offset if left else -lat_offset
        viewer_yaw = yaw + 90 if left else yaw - 90

        # 计算相机位置
        x = hero_transform.location.x + lon_offset * np.cos(yaw_rad) + lat_offset * np.sin(yaw_rad)
        y = hero_transform.location.y + lon_offset * np.sin(yaw_rad) - lat_offset * np.cos(yaw_rad)

        # 生成相机的Transform
        return carla.Transform(carla.Location(x=x, y=y, z=1.0),
                               carla.Rotation(pitch=0, yaw=viewer_yaw, roll=0))


def bev_transform(height=50., lon_offset=0., lat_offset=0., absolute=False, hero_transform=None):
    # 对于lat offset 车左为正方向。对于lon offset 车前为正方向。
    # 若absolute为True，则返回的transform为世界坐标系，否则为hero的自车坐标系下的相对transform。
    if absolute:
        assert hero_transform is not None, "hero_transform must be provided for absolute BEV transform"
        yaw = hero_transform.rotation.yaw
        yaw_rad = np.radians(yaw)
        x = hero_transform.location.x + lon_offset * np.cos(yaw_rad) + lat_offset * np.sin(yaw_rad)
        y = hero_transform.location.y + lon_offset * np.sin(yaw_rad) - lat_offset * np.cos(yaw_rad)
        return carla.Transform(carla.Location(x=x, y=y, z=height),
                                carla.Rotation(pitch=-90, yaw=yaw, roll=0))
    else:
        return carla.Transform(carla.Location(x=lon_offset, y=lat_offset, z=height),
                               carla.Rotation(pitch=-90, yaw=0, roll=0))

#################### weather ############################
def generate_random_weather(preset_combination=True):
    '''
    if preset_combination, then the weather will be chosen from the combination of presets
    otherwise, it will be randomly chosen from all possible weather value
    '''
    if preset_combination:
        weather = carla.WeatherParameters()
        sun_str = random.choice(list(SUN_PRESETS))
        weather_str = random.choice(list(WEATHER_PRESETS))
        weather = modify_weather_param(weather, sun_str, weather_str)
    else:
        weather = carla.WeatherParameters(
            cloudiness=np.random.randint(80),
            precipitation=np.random.choice(list(carla.WeatherParameters.Precipitation)),
            precipitation_deposits=np.random.choice(list(carla.WeatherParameters.PrecipitationDeposits)),
            wind_intensity=np.random.randint(100),
            sun_azimuth_angle=np.random.randint(180),
            sun_altitude_angle=np.random.randint(-89, 89),
            fog_density=np.random.randint(100),
            fog_distance=np.random.randint(50, 500),
            fog_falloff=np.random.exponential(scale=1/1.0),
            wether_persistence=np.random.randint(1000),
            wetness=np.random.randint(100)
        )
    return weather

def modify_weather_param(current_weather, sun:Union[str, Sequence, None]=None,
                         weather:Union[str, Sequence, None]=None):
    if sun is not None:
        if isinstance(sun, str):
            assert sun in SUN_PRESETS, "sun must be in {}".format(SUN_PRESETS)
            sun_param = SUN_PRESETS[sun]
        else:
            assert len(sun) == 2, "sun param must be of length 2"
            sun_param = sun
        current_weather.sun_altitude_angle = sun_param[0]
        current_weather.sun_azimuth_angle = sun_param[1]

    if weather is not None:
        if isinstance(weather, str):
            assert weather in WEATHER_PRESETS, "weather must be in {}".format(WEATHER_PRESETS)
            weather_param = WEATHER_PRESETS[weather]
        else:
            assert len(weather) == 11, "weather param must be of length 11."
            weather_param = weather
        current_weather.cloudiness = weather_param[0]
        current_weather.precipitation = weather_param[1]
        current_weather.precipitation_deposits = weather_param[2]
        current_weather.wind_intensity = weather_param[3]
        current_weather.fog_density = weather_param[4]
        current_weather.fog_distance = weather_param[5]
        current_weather.fog_falloff = weather_param[6]
        current_weather.wetness = weather_param[7]
        current_weather.scattering_intensity = weather_param[8]
        current_weather.mie_scattering_scale = weather_param[9]
        current_weather.rayleigh_scattering_scale = weather_param[10]
    return current_weather

################## autopilot ###########################
def set_autopilot(vehicle_actor, enable=True):
    vehicle_actor.set_autopilot(enable)


################## car light ############################
def set_autolight(vehicle_actor, traffic_manager, enable=True):
    # if vehicle_actor.type_id.startswith("vehicle"):
    traffic_manager.update_vehicle_lights(vehicle_actor, enable)


def set_car_light(actor, light_state: carla.VehicleLightState):
    if actor.type_id.startswith("vehicle"):
        actor.set_light_state(light_state)


def get_car_light(actor) -> carla.VehicleLightState:
    if actor.type_id.startswith("vehicle"):
        light_state = actor.get_light_state()
    else:
        light_state = carla.VehicleLightState.NONE
    return light_state

########################################

def modify_vehicle_physics(vehicle_actor):
    #If actor is not a vehicle, we cannot use the physics control
    try:
        physics_control = vehicle_actor.get_physics_control() # get the last physics control applied to the vehicle
        physics_control.use_sweep_wheel_collision = True
        vehicle_actor.apply_physics_control(physics_control)
    except Exception:
        pass
