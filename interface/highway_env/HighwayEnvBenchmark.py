'''
闭环benchmark
'''
import gymnasium as gym
# import gym
import highway_env
from gymnasium.wrappers import RecordVideo
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.graphics import VehicleGraphics

import pygame
import warnings
import random
import numpy as np

import spider
from spider.interface.BaseBenchmark import BaseBenchmark
from spider.interface.highway_env.HighwayEnvInterface import HighwayEnvInterface


class HighwayEnvBenchmark(BaseBenchmark):
    def __init__(self, dt=0.1, config=None):
        if config and "env_config" in config and "observation" in config["env_config"]:
            # obs_config = config["env_config"]["observation"]
            # if not (obs_config["absolute"] and not obs_config["normalize"]) :
            warnings.warn("You should set absolute=True,normalize=False.")
        super(HighwayEnvBenchmark, self).__init__(config)
        self.config["env_config"]["policy_frequency"] = int(1/dt)
        # self.config["env_config"]["offscreen_rendering"]: True

        self.interface:HighwayEnvInterface = None

        random.seed(self.config["random_seed"]) # 现在这个随机不对
        np.random.seed(self.config["random_seed"])
        self.save_video = self.config["save_video"]
        self.video_root = self.config["video_root"]
        self.metrics = {
            "avg_cumulative_reward": 0.0
        }

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        config = super().default_config()
        config.update({
            "env_name": "highway-v0",
            "max_steps": 100,
            "episode_num": 1,
            "env_config": {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy", "heading"],
                    "absolute": True,
                    "normalize": False  # 这两条必须要加上
                },
                "action": {
                    "type": "ContinuousAction",
                },
                "policy_frequency": 10,
                # "offscreen_rendering": True
            }
        })
        return config

    def initial_environment(self):
        '''
        根据给定的config，初始化环境self._env
        '''
        # 'render_modes': ['human', 'rgb_array'], # human是只渲染，rgb——array是渲染函数会返回一个rgb图像
        if self.env is None:
            self.env = gym.make(self.config['env_name'],render_mode='rgb_array')

        self.env.configure(self.config["env_config"])
        if self.save_video:
            self.env = RecordVideo(self.env, video_folder=self.video_root, episode_trigger=lambda e: True)
            self.env.unwrapped.set_record_video_wrapper(self.env)
        
        self.interface = HighwayEnvInterface(self.env)
        return self.interface.reset()

    def test(self, spider_planner, show_video: bool = False, save_video: bool = True):
        '''
        给定一个planner，在设置好的环境里面开一遍，返回config中指定的metrics
        '''
        obs, info = self.initial_environment()

        for i in range(self.config["episode_num"]):
            self.interface.reset()
            self.env.render() # 为了初始化env.viewer, 否则为None

            for _ in range(self.config["max_steps"]):
                ego_veh_state, perception, local_map = self.interface.wrap_observation(obs)
                output = spider_planner.plan(ego_veh_state, perception, local_map)
                if output is None:
                    break # 没有可行解了，其实应该把reward给个低值。
                action = self.interface.convert_to_action(output)
                obs, reward, done, truncated, info = self.env.step(action)


                self.env.render()
                self.visualize_plan(output)

                if done or truncated:
                    break
        # todo:录像功能现在有问题 & metrics
        self.env.close()
        self.metrics["avg_cumulative_reward"] /= self.config["episode_num"]

    def update_metrics(self, reward):
        self.metrics["avg_cumulative_reward"] += reward

    def visualize_plan(self, planner_output):
        '''
        把规划结果画在仿真器渲染的画面上
        '''
        if self.interface.output_flag == spider.OUTPUT_TRAJECTORY:# 检查输出标志，确保是轨迹输出
            traj: spider.elements.Trajectory = planner_output
            # 获取仿真器画布和offscreen flag
            surface = self.env.viewer.sim_surface
            offscreen = self.env.viewer.offscreen

            traj_points_px = [[*surface.pos2pix(traj.x[0], traj.y[0])]] # 在像素坐标系中的轨迹点，先加入第一个轨迹点（当前点）
            for i in range(1, traj.steps):
                env_veh = Vehicle(None, position=[traj.x[i], traj.y[i]], heading=traj.heading[i], speed=traj.v[i])
                env_veh.action = {'steering': traj.steer[i], 'acceleration': traj.a[i]}
                VehicleGraphics.display(env_veh, surface, transparent=True, offscreen=offscreen)

                traj_points_px.append([*surface.pos2pix(env_veh.position[0], env_veh.position[1])])

            if len(traj_points_px) > 1:
                pygame.draw.lines(surface, (255, 0, 0), False, traj_points_px, 2)


        else:
            warnings.warn("can't visualize control output yet...")

        # 将画布内容绘制到屏幕上并刷新显示
        self.env.viewer.screen.blit(surface, (0, 0))
        pygame.display.flip()


    # @classmethod
    # def display(cls, vehicle: Vehicle, surface: "WorldSurface",
    #             transparent: bool = False,
    #             offscreen: bool = False,
    #             label: bool = False,
    #             draw_roof: bool = False) -> None:
    #     """
    #     Display a vehicle on a pygame surface.
    #
    #     The vehicle is represented as a colored rotated rectangle.
    #
    #     :param vehicle: the vehicle to be drawn
    #     :param surface: the surface to draw the vehicle on
    #     :param transparent: whether the vehicle should be drawn slightly transparent
    #     :param offscreen: whether the rendering should be done offscreen or not
    #     :param label: whether a text label should be rendered
    #     :param draw_roof: whether to draw the vehicle roof
    #     """
    #     if not surface.is_visible(vehicle.position):
    #         return  # 如果车辆位置在窗口外部，不进行绘制
    #
    #     v = vehicle
    #     tire_length, tire_width = 1, 0.3
    #     headlight_length, headlight_width = 0.72, 0.6
    #     roof_length, roof_width = 2.0, 1.5
    #
    #     # Vehicle rectangle
    #     length = v.LENGTH + 2 * tire_length
    #     # 创建车辆矩形的 Surface
    #     vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)),
    #                                      flags=pygame.SRCALPHA)  # 使用 per-pixel alpha
    #     rect = (surface.pix(tire_length),
    #             surface.pix(length / 2 - v.WIDTH / 2),
    #             surface.pix(v.LENGTH),
    #             surface.pix(v.WIDTH))
    #     rect_headlight_left = (surface.pix(tire_length + v.LENGTH - headlight_length),
    #                            surface.pix(length / 2 - (1.4 * v.WIDTH) / 3),
    #                            surface.pix(headlight_length),
    #                            surface.pix(headlight_width))
    #     rect_headlight_right = (surface.pix(tire_length + v.LENGTH - headlight_length),
    #                             surface.pix(length / 2 + (0.6 * v.WIDTH) / 5),
    #                             surface.pix(headlight_length),
    #                             surface.pix(headlight_width))
    #     color = cls.get_color(v, transparent)
    #     # 绘制车辆矩形和前灯
    #     pygame.draw.rect(vehicle_surface, color, rect, 0)
    #     pygame.draw.rect(vehicle_surface, cls.lighten(color), rect_headlight_left, 0)
    #     pygame.draw.rect(vehicle_surface, cls.lighten(color), rect_headlight_right, 0)
    #     if draw_roof:
    #         # 如果指定了绘制车顶，绘制车顶
    #         rect_roof = (surface.pix(v.LENGTH / 2 - tire_length / 2),
    #                      surface.pix(0.999 * length / 2 - 0.38625 * v.WIDTH),
    #                      surface.pix(roof_length),
    #                      surface.pix(roof_width))
    #         pygame.draw.rect(vehicle_surface, cls.darken(color), rect_roof, 0)
    #     pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)
    #
    #     # Tires
    #     if type(vehicle) in [Vehicle, BicycleVehicle]:
    #         # 绘制车轮
    #         tire_positions = [[surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
    #                           [surface.pix(tire_length), surface.pix(length / 2 + v.WIDTH / 2)],
    #                           [surface.pix(length - tire_length), surface.pix(length / 2 - v.WIDTH / 2)],
    #                           [surface.pix(length - tire_length), surface.pix(length / 2 + v.WIDTH / 2)]]
    #         tire_angles = [0, 0, v.action["steering"], v.action["steering"]]
    #         for tire_position, tire_angle in zip(tire_positions, tire_angles):
    #             # 创建车轮的 Surface
    #             tire_surface = pygame.Surface((surface.pix(tire_length), surface.pix(tire_length)), pygame.SRCALPHA)
    #             rect = (
    #             0, surface.pix(tire_length / 2 - tire_width / 2), surface.pix(tire_length), surface.pix(tire_width))
    #             pygame.draw.rect(tire_surface, cls.BLACK, rect, 0)
    #             # 旋转并绘制车轮
    #             cls.blit_rotate(vehicle_surface, tire_surface, tire_position, np.rad2deg(-tire_angle))
    #
    #     # Centered rotation
    #     h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
    #     position = [*surface.pos2pix(v.position[0], v.position[1])]
    #     if not offscreen:
    #         # 在非离屏模式下，使用 convert_alpha 以避免错误
    #         vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
    #     # 旋转并绘制整个车辆
    #     cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))
    #
    #     # Label
    #     if label:
    #         # 如果指定了绘制标签，绘制标签
    #         font = pygame.font.Font(None, 15)
    #         text = "#{}".format(id(v) % 1000)
