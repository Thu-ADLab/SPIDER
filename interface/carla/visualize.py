import pygame
import os
import math
import datetime
import weakref

import carla

from spider.interface.carla.common import *
from spider.interface.carla.presets import viewer_sensor_presets

# Viewer to visualize the image

class Viewer:
    # _available_views = ["first_person", "third_person", "bird_eye", "left_side", "right_side"]

    _views_relative_transform = {
        "first_person": first_person_view_transform(),
        "third_person": third_person_view_transform(),
        "bird_eye": bev_transform(),
        "left_side": side_view_transform(left=True),
        "right_side": side_view_transform(left=False)
    }

    # _default_image_size = (1280, 720)
    # _default_lidar_range = 80

    _sensor_presets = viewer_sensor_presets

    _attachment_type = carla.AttachmentType.Rigid

    def __init__(self, viewed_object:carla.Actor=None, sensor_type="camera_rgb", view="third_person",
                 recording=False, image_size=(1280, 720), lidar_range=80):
        self.sensor = None
        self.sensor_type = None
        self.view = view
        self.viewed_object = None

        self.surface = None
        self.recording = recording
        self.image_size = image_size
        self.lidar_range = lidar_range

        self.image_array = None


        if viewed_object is None:
            print("No viewed object specified, cannot spawn sensor.")
        else:
            self.spawn_sensor(viewed_object, sensor_type, view)


    @property
    def sensor_type_info(self):
        return self._sensor_presets[self.sensor_type]

    def spawn_sensor(self, object:carla.Actor, sensor_type="camera_rgb", view="third_person"):
        '''
        Spawn a sensor and attach it to the object.
        :param object: The object to attach the sensor to.
        :param sensor_type: The type of sensor to spawn.
        :param view: The view of the sensor.
        '''
        assert sensor_type in self._sensor_presets, "Sensor type must be one of {}".format(self._sensor_presets.keys())
        assert view in self._views_relative_transform, "View must be one of {}".format(self._views_relative_transform.keys())

        self.sensor_type = sensor_type
        self.view = view
        self.viewed_object = object

        if self.sensor is not None:
            print("Found existing sensor, destroy it.")
            self.destroy()

        world = object.get_world()
        # find the blueprint
        bp_name = self.sensor_type_info[0]
        bp = world.get_blueprint_library().find(bp_name)
        if bp_name.startswith('sensor.camera'):
            bp.set_attribute('image_size_x', str(self.image_size[0]))
            bp.set_attribute('image_size_y', str(self.image_size[1]))
        elif bp_name.startswith('sensor.lidar'):
            bp.set_attribute('range', str(self.lidar_range))
        # the transform for the certain view
        tf = self._views_relative_transform[view]

        self.sensor = world.spawn_actor(
            bp, tf,
            attach_to=object,
            attachment_type=self._attachment_type
        )

        weak_self = weakref.ref(self) # pass the lambda a weak reference to self to avoid circular reference.
        self.sensor.listen(lambda image: Viewer._parse_image(weak_self, image))

        print("Viewer spawned: ", self.sensor_type_info[2])

    def set_image_size(self, image_size_x, image_size_y):
        self.image_size = (image_size_x, image_size_y)

    def set_lidar_range(self, lidar_range):
        self.lidar_range = lidar_range

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        print('Recording %s' % ('On' if self.recording else 'Off'))

    def change_view(self, view):
        assert view in self._views_relative_transform, "View must be one of {}".format(
            self._views_relative_transform.keys())
        if self.viewed_object is None or self.sensor_type is None:
            print("No sensor to change view.")
        else:
            self.spawn_sensor(self.viewed_object, self.sensor_type, view)
            print("Viewer view changed to: ", view)

    def change_sensor(self, sensor_type):
        assert sensor_type in self._sensor_presets, "Sensor type must be one of {}".format(self._sensor_presets.keys())
        if self.viewed_object is None or self.view is None:
            print("No sensor to change")
        else:
            self.spawn_sensor(self.viewed_object, sensor_type, self.view)
            print("Viewer sensor changed to: ", self.sensor_type_info[2])

    def render(self, display=None):
        """Render method"""
        if display is None:
            display = pygame.display.set_mode(self.image_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
        return display


    def destroy(self):
        self.sensor.destroy()
        self.sensor = None
        self.surface = None
        self.sensor_type = None
        self.image_array = None

    # @property
    # def available_views:

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        # 对于传感器的判断变一下
        if self.sensor_type_info[0].startswith('sensor.lidar'):
            # # print("!!!!!!!!!!!!!!!", self.image_size)
            # points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            # points = np.reshape(points, (int(points.shape[0] / 4), 4))
            # lidar_data = np.array(points[:, :2])
            # # print(lidar_data)
            # lidar_data *= min(self.image_size) / (2*self.lidar_range) # 100.0
            # lidar_data += (0.5 * self.image_size[1], 0.5 * self.image_size[0])
            # lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            # lidar_data = lidar_data.astype(np.int32)
            # lidar_data = np.reshape(lidar_data, (-1, 2))
            # lidar_img_size = (self.image_size[1], self.image_size[0], 3)
            # lidar_img = np.zeros(lidar_img_size)
            # lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            # self.image_array = lidar_img
            # self.surface = pygame.surfarray.make_surface(lidar_img)


            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            # print(lidar_data)
            lidar_data *= min(self.image_size) / (2 * self.lidar_range)  # 100.0
            lidar_data += (0.5 * self.image_size[0], 0.5 * self.image_size[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.image_size[0], self.image_size[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.image_array = lidar_img
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensor_type_info[1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.image_array = array  # qzl: 保存一波
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)






















def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- HUD ------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        # pygame.draw.line(display, (255,0,0), (300,0), (300,600), 2) # qzl: 画直线分割线，表示BEV用的
        # pygame.draw.line(display, (255,0,0), (0,300), (1300,300), 2) # qzl
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ------------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)
