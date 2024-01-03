import numpy as np

N = steps = 50

dt = 0.1

Ms = np.zeros((N,2*N))
Ms[:,:N] = np.eye(N)

Ml = np.zeros((N,2*N))
Ml[:,N:] = np.eye(N)

Minus = np.zeros(N)
Minus[0] = -1
Minus[-1] = 1

Diff = np.zeros((N-1,N))
for i in range(N-1):
    Diff[i,i] = -1
    Diff[i, i+1] = 1
Diff /= dt

Appe = np.zeros((N,N-1))
Appe[1:,:] = np.eye(N-1)

First = np.zeros(N)
First[0] = 1


# Final = np.zeros((N,1))
Final = np.zeros(N)
Final[-1] = 1
# Final = np.reshape(Final,(N))

G1 = Appe@Diff # 1阶导公式中x的参数矩阵
H1_1 = First # 1阶导公式中1阶导初始值的参数矩阵

G2 = G1@G1 # 2阶导公式中x的参数矩阵
H2_1 = G1@H1_1 # 2阶导公式中1阶导初始值的参数矩阵
H2_2 = First # 2阶导公式中2阶导初始值的参数矩阵

G3 = Diff@G2 # 2阶导公式中x的参数矩阵
H3_1 = Diff@H2_1 # 3阶导公式中1阶导初始值的参数矩阵
H3_2 = Diff@H2_2 # 3阶导公式中2阶导初始值的参数矩阵

weight_comf = 0.1
weight_jerk_s = 1
weight_jerk_l = 10

weight_eff = 2
weight_eff_s = 1
weight_eff_s_d = 0.5

weight_safe = 2
weight_safe_lateral_offset = 1


class Objective:
    def __init__(self,f=None,Q=None,func=None):
        self.f = f
        self.Q = Q
        self.func = func

    def linear(self):
        return self.f

    def quadratic(self):
        return self.Q, self.f

    def function(self):
        return self.func

    # def all(self):
    #     return self.Q, self.f, self.func


class Constraints:
    def __init__(self, A=None, b=None, Aineq=None, bineq=None, conditions=None):
        self.A = A
        self.b = b
        self.Aineq = Aineq
        self.bineq = bineq
        self.cond = conditions

    def linear(self):
        return self.A, self.b, self.Aineq, self.bineq

    def conditions(self):
        return self.cond





if __name__ == '__main__':
    x = np.arange(0,50,1)
    x_dot = G1@x + H1_1*100
    print(x_dot)
    pass