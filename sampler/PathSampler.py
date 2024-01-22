'''
including some samplers that generates a 2D curve as path
'''
import numpy as np
from spider.sampler.BaseSampler import BaseSampler
from spider.elements.curves import ParametricCurve, BezierCurve
from spider.utils.transform import FrenetCoordinateTransformer

class BezierCurveSampler(BaseSampler):
    def __init__(self, end_s_candidates, end_l_candidates, num_control_points=5, max_transition_length=5.0):
        '''
        end_dx_candidates: x一阶导的终值候选项
        '''
        super(BezierCurveSampler, self).__init__()
        assert num_control_points == 5, "Only support 5 points sampling for now..."
        self.end_s_candidates = end_s_candidates
        self.end_l_candidates = end_l_candidates
        self.num_control_points = num_control_points
        self.max_transition_length = max_transition_length #transition_length是为了保证初始yaw和末尾yaw而延申的长度

    def sample(self, start_x, start_y, start_yaw, frenet_transformer: FrenetCoordinateTransformer):
        '''
        referline应该是参数化曲线，输入s,输出x,y
        '''

        start_s = frenet_transformer.cart2frenet(start_x, start_y,order=0).s

        samples = []
        for delta_s in self.end_s_candidates:
            end_s = delta_s+start_s
            for end_l in self.end_l_candidates:
                end_x, end_y, end_yaw = self._calc_pose(end_s,end_l, frenet_transformer.refer_line_csp)
                control_points = self._calc_control_points(start_x, start_y, start_yaw, end_x, end_y, end_yaw)
                samples.append(BezierCurve(control_points))

        return samples

    def _calc_control_points(self, start_x, start_y, start_yaw, end_x, end_y, end_yaw):
        '''
        only support 5 points
        '''
        dist = np.array((end_y-start_y)**2 + (end_x-start_x)**2)
        transition_length = min([dist/(self.num_control_points-1), self.max_transition_length])
        control_points = np.array([
            [start_x, start_y],
            [start_x + transition_length*np.cos(start_yaw), start_y + transition_length*np.sin(start_yaw)],

            [end_x - transition_length*np.cos(end_yaw), end_y - transition_length*np.sin(end_yaw)],
            [end_x, end_y]
        ])
        control_points = np.insert(control_points, 2, (control_points[1]+control_points[-2])/2, axis=0)
        return control_points


    # def _control_points_sl(self, start_s, end_s, start_l, end_l):
    #     ss = np.linspace(start_s, end_s, self.num_control_points)
    #     ls = np.array([start_l, start_l, 0.5*(start_l+end_l), end_l, end_l])
    #     return ss, ls


    def _calc_pose(self, s, l, reference_line_curve:ParametricCurve):
        # todo: 用frenet transformer替代
        x, y = reference_line_curve(s)
        theta = reference_line_curve.calc_yaw(s)
        if np.all(l == 0):
            return x,y, theta
        else:
            x = x - l * np.sin(theta)
            y = y + l * np.cos(theta)
            return x, y, theta

