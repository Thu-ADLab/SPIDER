import numpy as np

from spider.elements.vector import rotate


class RelativeCoordinateTransformer:
    def __init__(self, ego_x=None, ego_y=None, ego_yaw=None, ego_vx=None, ego_vy=None):
        self.ego_x = 0.
        self.ego_y = 0.
        self.ego_yaw = 0.
        self.ego_vx = 0.
        self.ego_vy = 0.

        self.set_ego_pose(ego_x, ego_y, ego_yaw)
        self.set_ego_velocity(ego_vx, ego_vy)

    def set_ego_pose(self, ego_x, ego_y, ego_yaw):
        self.ego_x = ego_x
        self.ego_y = ego_y
        self.ego_yaw = ego_yaw

    def set_ego_velocity(self, ego_vx, ego_vy):
        self.ego_vx = ego_vx
        self.ego_vy = ego_vy

    def abs2rel(self, x, y, yaw=None, vx=None, vy=None, ego_pose=None, ego_velocity=None):
        '''
        ego_pose: (ego_x, ego_y, ego_yaw)
        '''
        if ego_pose is not None:
            self.set_ego_pose(*ego_pose)
        if ego_velocity is not None:
            self.set_ego_velocity(*ego_velocity)

        delta_vec = np.array([x - self.ego_x, y - self.ego_y]).T
        rel_x, rel_y = rotate(delta_vec, (0, 0), -self.ego_yaw).T

        rel_yaw = yaw - self.ego_yaw if yaw is not None else None

        if vx is None or vy is None:
            rel_vx = rel_vy = None
        else:
            delta_vel_vec = np.array([vx - self.ego_vx, vy - self.ego_vy]).T
            rel_vx, rel_vy = rotate(delta_vel_vec, (0, 0), -self.ego_yaw).T

        return rel_x, rel_y, rel_yaw, rel_vx, rel_vy


    def rel2abs(self, rel_x, rel_y, rel_yaw=None, rel_vx=None, rel_vy=None, ego_pose=None, ego_velocity=None):
        '''
        ego_pose: (ego_x, ego_y, ego_yaw)
        '''
        if ego_pose is not None:
            self.set_ego_pose(*ego_pose)
        if ego_velocity is not None:
            self.set_ego_velocity(*ego_velocity)

        delta_vec = rotate(np.array([rel_x, rel_y]), (0, 0), self.ego_yaw).T
        abs_x = self.ego_x + delta_vec[..., 0]
        abs_y = self.ego_y + delta_vec[..., 1]

        abs_yaw = self.ego_yaw + rel_yaw if rel_yaw is not None else None

        if rel_vx is None or rel_vy is None:
            abs_vx = abs_vy = None
        else:
            delta_vel_vec = rotate(np.array([rel_vx, rel_vy]), (0, 0), self.ego_yaw).T
            abs_vx = self.ego_vx + delta_vel_vec[...,0]
            abs_vy = self.ego_vy + delta_vel_vec[...,1]

        return abs_x, abs_y, abs_yaw, abs_vx, abs_vy

if __name__ == '__main__':
    tf = RelativeCoordinateTransformer()
    tf.abs2rel(2, 2, 3.14 / 3, 1, 1, (1, 1, 3.14 / 6), (1, 0))
    print(tf)
