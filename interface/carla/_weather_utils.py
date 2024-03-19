import carla
import sys

SUN_PRESETS = {
    'day': (45.0, 0.0),
    'night': (-90.0, 0.0),
    'sunset': (0.5, 0.0)
}

WEATHER_PRESETS = {
    'clear': [10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0331],
    'overcast': [80.0, 0.0, 0.0, 50.0, 2.0, 0.75, 0.1, 10.0, 0.0, 0.03, 0.0331],
    'rain': [100.0, 80.0, 90.0, 100.0, 7.0, 0.75, 0.1, 100.0, 0.0, 0.03, 0.0331]}

CAR_LIGHTS = {
    'None' : [carla.VehicleLightState.NONE],
    'Position' : [carla.VehicleLightState.Position],
    'LowBeam' : [carla.VehicleLightState.LowBeam],
    'HighBeam' : [carla.VehicleLightState.HighBeam],
    'Brake' : [carla.VehicleLightState.Brake],
    'RightBlinker' : [carla.VehicleLightState.RightBlinker],
    'LeftBlinker' : [carla.VehicleLightState.LeftBlinker],
    'Reverse' : [carla.VehicleLightState.Reverse],
    'Fog' : [carla.VehicleLightState.Fog],
    'Interior' : [carla.VehicleLightState.Interior],
    'Special1' : [carla.VehicleLightState.Special1],
    'Special2' : [carla.VehicleLightState.Special2],
    'All' : [carla.VehicleLightState.All]}

LIGHT_GROUP = {
    'None' : [carla.LightGroup.NONE],
    # 'Vehicle' : [carla.LightGroup.Vehicle],
    'Street' : [carla.LightGroup.Street],
    'Building' : [carla.LightGroup.Building],
    'Other' : [carla.LightGroup.Other]}

def apply_sun_presets(args, weather):
    """Uses sun presets to set the sun position"""
    if args.sun is not None:
        if args.sun in SUN_PRESETS:
            weather.sun_altitude_angle = SUN_PRESETS[args.sun][0]
            weather.sun_azimuth_angle = SUN_PRESETS[args.sun][1]
        else:
            print("[ERROR]: Command [--sun | -s] '" + args.sun + "' not known")
            sys.exit(1)


def apply_weather_presets(args, weather):
    """Uses weather presets to set the weather parameters"""
    if args.weather is not None:
        if args.weather in WEATHER_PRESETS:
            weather.cloudiness = WEATHER_PRESETS[args.weather][0]
            weather.precipitation = WEATHER_PRESETS[args.weather][1]
            weather.precipitation_deposits = WEATHER_PRESETS[args.weather][2]
            weather.wind_intensity = WEATHER_PRESETS[args.weather][3]
            weather.fog_density = WEATHER_PRESETS[args.weather][4]
            weather.fog_distance = WEATHER_PRESETS[args.weather][5]
            weather.fog_falloff = WEATHER_PRESETS[args.weather][6]
            weather.wetness = WEATHER_PRESETS[args.weather][7]
            weather.scattering_intensity = WEATHER_PRESETS[args.weather][8]
            weather.mie_scattering_scale = WEATHER_PRESETS[args.weather][9]
            weather.rayleigh_scattering_scale = WEATHER_PRESETS[args.weather][10]
        else:
            print("[ERROR]: Command [--weather | -w] '" + args.weather + "' not known")
            sys.exit(1)


def apply_weather_values(args, weather):
    """Set weather values individually"""
    if args.azimuth is not None:
        weather.sun_azimuth_angle = args.azimuth
    if args.altitude is not None:
        weather.sun_altitude_angle = args.altitude
    if args.clouds is not None:
        weather.cloudiness = args.clouds
    if args.rain is not None:
        weather.precipitation = args.rain
    if args.puddles is not None:
        weather.precipitation_deposits = args.puddles
    if args.wind is not None:
        weather.wind_intensity = args.wind
    if args.fog is not None:
        weather.fog_density = args.fog
    if args.fogdist is not None:
        weather.fog_distance = args.fogdist
    if args.fogfalloff is not None:
        weather.fog_falloff = args.fogfalloff
    if args.wetness is not None:
        weather.wetness = args.wetness
    if args.scatteringintensity is not None:
        weather.scattering_intensity = args.scatteringintensity
    if args.miescatteringscale is not None:
        weather.mie_scattering_scale = args.miescatteringscale
    if args.rayleighscatteringscale is not None:
        weather.rayleigh_scattering_scale = args.rayleighscatteringscale