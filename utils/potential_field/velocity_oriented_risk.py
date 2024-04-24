
import numpy as np


G, K_risk = 9, 0.3
Ms = {i:1 for i in range(50)} # qzl: a ratio determined by the class of boundary points
max_risk = 1.0
_epsilon = 0.001

def vel_oriented_risk(target_xs, target_ys, obs_x, obs_y, obs_vx, obs_vy, obs_class=0, ego_radius=1.0):
    dx = target_xs - obs_x
    dy = target_ys - obs_y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # if np.any(dist <= ego_radius):
    #     return max_risk # *10 collision!!

    rr = (dist - ego_radius) ** 2 # dist < ego radius的在后面会被置为max risk
    # rr = dist ** 2
    rr[rr < _epsilon] = _epsilon # avoid division by zero

    v = np.sqrt(obs_vx ** 2 + obs_vy ** 2)
    vtheta = np.arctan2(obs_vy, obs_vx)  # if obvx != 0 else 0.5*np.pi*obvy/abs(obvy)
    rtheta = np.arctan2(dy, dx)
    theta = vtheta - rtheta
    M = Ms.get(int(obs_class), 1.0)
    risk = np.exp(K_risk * v * np.cos(theta)) * G * M / rr
    # risk = np.exp(K_risk * v * np.cos(theta) * 0.5 * (1 + np.cos(2 * theta))) * G * M[int(class_)]/ rr

    risk[dist<= ego_radius] = max_risk

    risk[risk > max_risk] = max_risk
    # risk = np.exp(k2*v*np.cos(theta)*(1 - np.abs(theta)//(2*np.pi))) * G / rr
    return risk

