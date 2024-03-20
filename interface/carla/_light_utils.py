import carla
from typing import Union
import numpy as np

from spider.interface.carla.presets import CARLIGHT_TO_LABEL


###### car light ########
def set_autolight(traffic_manager, actors):
    for actor in actors:
        if actor.type_id.startswith("vehicle"):
            traffic_manager.update_vehicle_lights(actor, True)


def set_car_light(actor, light_state: carla.VehicleLightState):
    if actor.type_id.startswith("vehicle"):
        actor.set_light_state(light_state)


def get_car_light(actor) -> carla.VehicleLightState:
    if actor.type_id.startswith("vehicle"):
        light_state = actor.get_light_state()
    else:
        light_state = carla.VehicleLightState.NONE
    return light_state

def decompose_car_light(car_light: Union[carla.VehicleLightState, int]):
    '''
    example:
    >>> veh.get_light_state()
    carla.libcarla.VehicleLightState(11)
    >>> decompose_car_light(veh.get_light_state())
    (array([0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), ['Position', 'LowBeam', 'Brake'])
    >>> decompose_car_light(11)
    (array([0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]), ['Position', 'LowBeam', 'Brake'])
    '''
    multi_hot = np.zeros(len(CARLIGHT_TO_LABEL), dtype=int)
    labels = []
    for i, light in enumerate(CARLIGHT_TO_LABEL):
        if light & car_light:
            multi_hot[i] = 1
            labels.append(CARLIGHT_TO_LABEL[light])
    return multi_hot, labels
