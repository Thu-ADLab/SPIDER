import numpy as np

# vector定义成形状为(n,)的一维ndarray，matrix定义为形状为(m,n)的二维ndarray

def vec_in(vector, matrix):
    '''
    判断 一个向量 是否在 一堆向量里面
    '''
    return (matrix == vector).all(1).any()


def find_vec(vector, matrix, find_all=False):
    '''
    在一堆向量里， 找到一个某个向量的索引， 并返回第一个
    '''
    idxs = np.where((matrix == vector).all(1))#[0]
    if find_all:
        return idxs
    else:
        if len(idxs) == 0:
            return None
        else:
            return idxs[0]

def normalize(vector: np.ndarray):
    '''
    求一个矢量方向的单位矢量
    :param vector: 任意维度向量
    :return: 单位向量
    '''
    # vector = np.array(vector)
    if modulus(vector) == 0:
        return np.zeros_like(vector)
    vector_norm = vector / modulus(vector)
    return vector_norm

def modulus(vector):
    return np.linalg.norm(vector)

def project(vector, segment_vector, calc_distance:bool=False):
    '''
    project vector or matrix to axis_vector
    :param array: vector or matrix
    :param segment_vector:
    :return: projection
    '''
    # 正值表示在目标向量的左侧（从目标向量到vector逆时针），负值表示在参考线的右侧（从目标向量到vector顺时针）。
    # 这种约定符合参考线在Frenet坐标系中的定义
    # 即：一般认为沿着参考线s增加方向的左边为正，右边为负
    segment_unit_vector = normalize(segment_vector)
    projection = np.dot(vector, normalize(segment_vector)) # projection length

    if calc_distance:
        distance_signed = np.cross(segment_unit_vector, vector)  # 叉乘
        return projection, float(distance_signed)
    else:
        return projection


def rotate90(vec:np.ndarray):
    '''
    逆时针旋转90度
    '''
    return np.flip(vec)*[1,-1]  # 离谱bug，1写成0了

def rotate(array, anchor, angle, clockwise=False):
    '''

    :param array: vector or 2-column matrix
    :param angle: rad required
    :param clockwise: default False
    :return:
    '''
    delta_array = array.copy()
    delta_array[:,0] -= anchor[0]
    delta_array[:, 1] -= anchor[1]
    if clockwise:
        angle = -angle
    rot = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    delta_array = delta_array @ rot
    delta_array[:, 0] += anchor[0]
    delta_array[:, 1] += anchor[1]
    return delta_array


class Vector3D:
    # 这边Vector的定义有歧义了，思考一下怎么修改
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


