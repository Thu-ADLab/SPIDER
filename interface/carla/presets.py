import carla

SUN_PRESETS = {
    'day': (45.0, 0.0),
    'night': (-90.0, 0.0),
    'sunset': (0.5, 0.0)}

WEATHER_PRESETS = {
    'clear': [10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0331],
    'overcast': [80.0, 0.0, 0.0, 50.0, 2.0, 0.75, 0.1, 10.0, 0.0, 0.03, 0.0331],
    'rain': [100.0, 80.0, 90.0, 100.0, 7.0, 0.75, 0.1, 100.0, 0.0, 0.03, 0.0331]}

LABEL_TO_CARLIGHT = {
    'None' : carla.VehicleLightState.NONE,
    'Position' : carla.VehicleLightState.Position,
    'LowBeam' : carla.VehicleLightState.LowBeam,
    'HighBeam' : carla.VehicleLightState.HighBeam,
    'Brake' : carla.VehicleLightState.Brake,
    'RightBlinker' : carla.VehicleLightState.RightBlinker,
    'LeftBlinker' : carla.VehicleLightState.LeftBlinker,
    'Reverse' : carla.VehicleLightState.Reverse,
    'Fog' : carla.VehicleLightState.Fog,
    'Interior' : carla.VehicleLightState.Interior,
    'Special1' : carla.VehicleLightState.Special1,
    'Special2' : carla.VehicleLightState.Special2,
    # 'All' : carla.VehicleLightState.All
}

CARLIGHT_TO_LABEL = {value: key for key, value in LABEL_TO_CARLIGHT.items()}

VEHICLES_WITH_LIGHT = [ # for carla 0.9.13
    "vehicle.chevrolet.impala",
    "vehicle.dodge.charger_police",
    "vehicle.audi.tt",
    "vehicle.mercedes.coupe",
    "vehicle.mercedes.coupe_2020",
    "vehicle.dodge.charger_2020",
    "vehicle.lincoln.mkz_2020",
    "vehicle.dodge.charger_police_2020",
    "vehicle.audi.etron",
    "vehicle.volkswagen.t2_2021",
    "vehicle.tesla.cybertruck",
    "vehicle.lincoln.mkz_2017",
    "vehicle.ford.mustang",
    "vehicle.volkswagen.t2",
    "vehicle.tesla.model3",
]


