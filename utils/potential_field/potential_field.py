import numpy as np
import matplotlib.pyplot as plt
# from pyheatmap.heatmap import HeatMap
# import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from spider.utils.potential_field.static_risk import static_risk
from spider.utils.potential_field.velocity_oriented_risk import vel_oriented_risk


class PotentialField:
    def __init__(self, obs_xs, obs_ys, obs_vxs, obs_vys, obs_classes, ego_radius,
                 risk_type="velocity_oriented"):
        self.obs_info = list(zip(obs_xs, obs_ys, obs_vxs, obs_vys, obs_classes))
        # self.k = 1  # 常数
        # self.a_b = 4  # 长短轴比例
        self.ego_radius = ego_radius
        # self.max_risk = 2

        self.risk_type = risk_type
        # self.build_grid()

    def build_grid(self, x_range, y_range, dx=0.5, dy=0.5, visualize=False):
        xs = np.arange(x_range[0], x_range[1], dx)
        ys = np.arange(y_range[0], y_range[1], dy)
        xxs, yys = np.meshgrid(xs, ys)
        xxs = xxs
        yys = yys
        risk = self.calc_risk_npts(xxs, yys)
        # risk[idx] = nocollision
        # risk[np.bitwise_not(idx)] = np.max(nocollision)*1
        # risk[risk>self.max_risk] = self.max_risk
        risk_grid_map = np.reshape(risk[::-1],(ys.shape[0],xs.shape[0]))
        if visualize:
            plt.imshow(risk_grid_map, extent=(np.amin(xs), np.amax(xs), np.amin(ys), np.amax(ys)),
                       cmap="Reds", norm=LogNorm())
            plt.colorbar()
        return risk_grid_map

    def show_grid(self, risk_grid_map, extent=None):
        plt.imshow(risk_grid_map, extent=extent, cmap="Reds", norm=LogNorm())
        plt.colorbar()

    def calc_risk_npts(self, xs, ys):

        # 用于计算一堆点的risk,返回一个数组
        risk_sum = np.zeros_like(xs,dtype=np.float32)

        for obs_x, obs_y, obs_vx, obs_vy, obs_class in self.obs_info:
            if self.risk_type == "static":
                risk= static_risk(xs,ys,obs_x,obs_y,obs_class,self.ego_radius)
            elif self.risk_type == "velocity_oriented":
                risk = vel_oriented_risk(xs,ys,obs_x,obs_y,obs_vx,obs_vy,obs_class,self.ego_radius)
            else:
                raise ValueError("Invalid risk type")
            risk_sum += risk
        return risk_sum


if __name__ == '__main__':
    pfield= PotentialField([0,10,40],[0,20,-10],[8,0,10],[4,-4,0],[0,0,0],1, )#"static"
    pfield.build_grid([-50,100],[-50,50],0.2,0.2, True)
    plt.show()

