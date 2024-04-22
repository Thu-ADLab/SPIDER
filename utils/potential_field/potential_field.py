import numpy as np
import matplotlib.pyplot as plt
# from pyheatmap.heatmap import HeatMap
# import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from spider.utils.potential_field.static_field import static_risk
from spider.utils.potential_field.velocity_oriented_risk import vel_oriented_risk



class potential_field:
    def __init__(self, obs_xs, obs_ys, obs_vxs, obs_vys, obs_classes, ego_radius):
        self.obs_info = list(zip(obs_xs, obs_ys, obs_vxs, obs_vys, obs_classes))
        # self.k = 1  # 常数
        # self.a_b = 4  # 长短轴比例
        self.ego_radius = ego_radius
        # self.max_risk = 2

        self.risk_type = "velocity_oriented"
        # self.build_grid()

    def build_grid(self, x_range, y_range, dx=0.5, dy=0.5):
        xs = np.arange(x_range[0], x_range[1],dx)
        ys = np.arange(y_range[0], y_range[1],dy)
        xxs, yys = np.meshgrid(xs, ys)
        # risk =  np.zeros_like(xxs,dtype=np.float32)
        # idx = yys < np.inf
        # for obx, oby, _, _,_ in self.obs_info:
        #     dist_2 = (xxs - obx)**2 + (yys - oby)**2
        #     idx = np.bitwise_and(dist_2 > self.ego_radius**2, idx)
        xxs = xxs#[idx] # 只取在构图空间内的
        yys = yys#[idx]
        risk = self.calc_risk_npts(xxs, yys) # nocollision =
        # risk[idx] = nocollision
        # risk[np.bitwise_not(idx)] = np.max(nocollision)*1
        # risk[risk>self.max_risk] = self.max_risk
        risk_grid_map = np.reshape(risk[::-1],(ys.shape[0],xs.shape[0]))
        plt.imshow(risk_grid_map, extent=(np.amin(xs), np.amax(xs), np.amin(ys), np.amax(ys)),
                   cmap="Reds", norm=LogNorm())
        plt.colorbar()

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # xxs, yys = np.meshgrid(xs, ys)
        # surf = ax.plot_surface(xxs,yys,risk,cmap=cm.coolwarm,linewidth=0)
        # fig.colorbar(surf,shrink=0.5,aspect=0.5)
        # plt.show()

    def calc_risk_npts(self, xs, ys):

        # if self.risk_type == "static":
        #     calc_risk = static_risk
        # elif self.risk_type == "velocity_oriented":
        #     calc_risk = vel_oriented_risk
        # else:
        #     raise ValueError("Invalid risk type")

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
    pfield= potential_field([0,10,40],[0,20,-10],[5,0,10],[8,-8,0],[0,0,0],1)
    pfield.build_grid([-50,100],[-50,50],0.2,0.2)
    plt.show()

