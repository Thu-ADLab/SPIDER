from typing import Callable

from spider.sampler.BaseSampler import BaseSampler
from spider.sampler.PolynomialSampler import QuinticPolyminalSampler, QuarticPolyminalSampler
from spider.sampler.Combiner import LatLonCombiner
from spider.utils.lane_decision import *

class LatticeSampler(BaseSampler):
    def __init__(self,
                 steps, dt, # trajectory parameters
                 end_T_candidates, end_v_candidates, # longitudinal sample
                 end_s_candidates, end_l_candidates, # lateral sample
                 lane_decision:Callable=None,  # 车道决策函数
                 calc_by_need=False # 如果为True，则惰性计算，返回的candidates只有在被索引时才会计算
                 ):
        super(LatticeSampler, self).__init__()
        self.steps = steps
        self.dt = dt
        self.end_T_candidates = end_T_candidates
        self.end_v_candidates = end_v_candidates
        self.end_s_candidates = end_s_candidates
        self.end_l_candidates = end_l_candidates
        self.num_samples = len(end_T_candidates) * len(end_v_candidates) * len(end_s_candidates) * len(end_l_candidates)

        self.lane_decision = NearestLaneDecision() if lane_decision is None else lane_decision

        self._calc_by_need = calc_by_need

        self.longitudinal_sampler = QuarticPolyminalSampler(self.end_T_candidates, self.end_v_candidates)
        self.lateral_sampler = QuinticPolyminalSampler(self.end_s_candidates, self.end_l_candidates)
        self.trajectory_combiner = LatLonCombiner(self.steps, self.dt)  # 默认路径-速度解耦的重新耦合


    def sample(self, lon_start_state, lat_start_state):
        '''
        sample a set of trajectory candidates
        :param lon_start_state: [s0, s_dot0, s_2dot0]
        :param lat_start_state: [l0, l_prime0, l_2prime0]
        '''

        lon_samples = self.longitudinal_sampler.sample(lon_start_state, self._calc_by_need)
        lat_samples = self.lateral_sampler.sample(lat_start_state, self._calc_by_need)
        candidate_trajectories = self.trajectory_combiner.combine(lat_samples, lon_samples, self._calc_by_need)
        return candidate_trajectories


    def sample_with_observation(self, ego_veh_state, perception, local_map, frenet2cart=False, cart_order=0):
        from spider.utils.transform.frenet import FrenetCoordinateTransformer

        target_lane_idx = self.lane_decision(ego_veh_state, perception, local_map)
        target_lane = local_map.lanes[target_lane_idx]

        coordinate_transformer = FrenetCoordinateTransformer()
        coordinate_transformer.set_reference_line(target_lane.centerline, target_lane.centerline_csp)

        fstate_start = coordinate_transformer.cart2frenet(ego_veh_state.x(), ego_veh_state.y(),
                                                          ego_veh_state.v(), ego_veh_state.yaw(),
                                                          ego_veh_state.a(), ego_veh_state.kappa(), order=2)
        candidate_trajectories = self.sample(
            (fstate_start.s, fstate_start.s_dot, fstate_start.s_2dot),
            (fstate_start.l, fstate_start.l_prime, fstate_start.l_2prime)
        )

        if not frenet2cart:
            return candidate_trajectories

        if not self._calc_by_need:
            candidate_trajectories = [coordinate_transformer.frenet2cart4traj(t, order=2)
                                             for t in candidate_trajectories]

        else:
            from spider.sampler.common import LazyList
            candidate_trajectories = LazyList(
                [LazyList.wrap_generator(coordinate_transformer.frenet2cart4traj, traj, order=cart_order)
                 for traj in candidate_trajectories]
            )
        return candidate_trajectories



    def sample_one(self, lon_start_state, lat_start_state, end_T, end_v, end_s, end_l):
        '''
        sample a set of trajectory candidates
        :param lon_start_state: [s0, s_dot0, s_2dot0]
        :param lat_start_state: [l0, l_prime0, l_2prime0]
        '''
        lon_sample = self.longitudinal_sampler.sample_one(lon_start_state, end_T, end_v)
        lat_sample = self.lateral_sampler.sample_one(lat_start_state, end_s, end_l)
        candidate_trajectory = self.trajectory_combiner.combine_one(lat_sample, lon_sample)
        return candidate_trajectory

    ################ setter ###################
    def set_end_T_candidates(self, end_T_candidates):
        self.end_T_candidates = end_T_candidates
        self.longitudinal_sampler = QuarticPolyminalSampler(self.end_T_candidates, self.end_v_candidates)

    def set_end_v_candidates(self, end_v_candidates):
        self.end_v_candidates = end_v_candidates
        self.longitudinal_sampler = QuarticPolyminalSampler(self.end_T_candidates, self.end_v_candidates)

    def set_end_s_candidates(self, end_s_candidates):
        self.end_s_candidates = end_s_candidates
        self.lateral_sampler = QuinticPolyminalSampler(self.end_s_candidates, self.end_l_candidates)

    def set_end_l_candidates(self, end_l_candidates):
        self.end_l_candidates = end_l_candidates
        self.lateral_sampler = QuinticPolyminalSampler(self.end_s_candidates, self.end_l_candidates)


if __name__ == '__main__':
    sampler = LatticeSampler(
             10, 0.1,  # trajectory parameters
             [2,5], [4,6],  # longitudinal sample
             [10,20], [-1,1],  # lateral sample
             calc_by_need=True
    )
    candidates = sampler.sample([0.,1.,0.,], [1.,0,0])
    for i in range(len(candidates)):
        print(i, candidates[i])
    # print(0,candidates[0])
    # print(1,candidates[1])
    # print(1,candidates[1])
    # print(2,candidates[2])
    # print(3,candidates[3])
    # print(4,candidates[4])
    pass
