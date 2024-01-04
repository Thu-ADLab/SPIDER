import numpy as np


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

    @property
    def is_quadratic(self):
        return (not (self.Q is None)) and (self.func is None)

    @property
    def is_linear(self):
        return (not (self.f is None)) and (self.Q is None) and (self.func is None)


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

    @property
    def is_linear(self) -> bool:
        return self.cond is None


class FrenetTrajOptimParam:
    def __init__(self, steps, dt):

        print('Building coefficient matrices and parameters for the optimizer in advance...')
        self.N = N = steps
        self.dt = dt

        self.ones_N = np.ones(N)
        self.ones_N_minus_1 = np.ones(N-1)
        # self.ones_2N = np.ones(2*N)
        self.zeros_2N_2N = np.zeros((2*N, 2*N))
        # se

        Ms = np.zeros((N, 2 * N))
        Ms[:, :N] = np.eye(N)
        self.Ms = Ms

        Ml = np.zeros((N, 2 * N))
        Ml[:, N:] = np.eye(N)
        self.Ml = Ml

        self.Minus = np.zeros(N)
        self.Minus[0] = -1
        self.Minus[-1] = 1 #最后一个值，和第一个值的差

        Diff = np.zeros((N - 1, N))
        for i in range(N - 1):
            Diff[i, i] = -1
            Diff[i, i + 1] = 1
        Diff /= dt
        self.Diff = Diff

        Appe = np.zeros((N, N - 1))
        Appe[1:, :] = np.eye(N - 1)
        self.Appe = Appe

        First = np.zeros(N)
        First[0] = 1
        self.First = First

        Final = np.zeros(N)
        Final[-1] = 1
        self.Final = Final

        self.G1   = G1 = Appe @ Diff  # 1阶导公式中x的参数矩阵
        self.H1_1 = H1_1 = First  # 1阶导公式中1阶导初始值的参数矩阵

        self.G2 = G2 = G1 @ G1  # 2阶导公式中x的参数矩阵
        self.H2_1 = G1 @ H1_1  # 2阶导公式中1阶导初始值的参数矩阵
        self.H2_2 = First  # 2阶导公式中2阶导初始值的参数矩阵

        self.G3 = G3 = Diff @ G2  # 2阶导公式中x的参数矩阵
        self.H3_1 = Diff @ self.H2_1  # 3阶导公式中1阶导初始值的参数矩阵
        self.H3_2 = Diff @ self.H2_2  # 3阶导公式中2阶导初始值的参数矩阵

        self.G2Ms, self.G2Ml = G2 @ Ms, G2 @ Ml
        self.G3Ms, self.G3Ml = G3 @ Ms, G3 @ Ml
        self.s_displacement = self.Minus @ Ms
        self.Diff_s = Diff @ Ms
        self.Diff_l = Diff @ Ml
        self.First_s = First @ Ms
        self.First_l = First @ Ml
        self.Final_s = Final @ Ms
        self.Final_l = Final @ Ml
        self.Final_s_dot = Final @ G1 @ Ms
        self.Final_s_2dot_coef = Final @ G2 @ Ms
        self.Final_s_2dot_bias = Final @ self.H2_1
        # comment: Final_s_dot @ X = final_s_dot
        # comment: Final_s_2dot_coef @ X + Final_s_2dot_bias * s_dot0 = final_s_2dot
        self.Final_l_dot = Final @ G1 @ Ml
        self.Final_l_2dot_coef = Final @ G2 @ Ml
        self.Final_l_2dot_bias = Final @ self.H2_1
        # comment: Final_l_dot @ X = final_l_dot
        # comment: Final_l_2dot_coef @ X + Final_l_2dot_bial * l_dot0 = final_l_2dot
        

        weight_comf = 0.1
        weight_jerk_s = 1
        weight_jerk_l = 10

        weight_eff = 2
        weight_eff_s = 1
        weight_eff_s_d = 0.5

        weight_safe = 2
        weight_safe_lateral_offset = 1
        print('Parameters Definition Completed. N=%d, dt=%.2f' % (N, dt))
        return

# N = steps = 50
#
# dt = 0.1
#
# Ms = np.zeros((N,2*N))
# Ms[:,:N] = np.eye(N)
#
# Ml = np.zeros((N,2*N))
# Ml[:,N:] = np.eye(N)
#
# Minus = np.zeros(N)
# Minus[0] = -1
# Minus[-1] = 1
#
# Diff = np.zeros((N-1,N))
# for i in range(N-1):
#     Diff[i,i] = -1
#     Diff[i, i+1] = 1
# Diff /= dt
#
# Appe = np.zeros((N,N-1))
# Appe[1:,:] = np.eye(N-1)
#
# First = np.zeros(N)
# First[0] = 1
#
#
# # Final = np.zeros((N,1))
# Final = np.zeros(N)
# Final[-1] = 1
# # Final = np.reshape(Final,(N))
#
# G1 = Appe@Diff # 1阶导公式中x的参数矩阵
# H1_1 = First # 1阶导公式中1阶导初始值的参数矩阵
#
# G2 = G1@G1 # 2阶导公式中x的参数矩阵
# H2_1 = G1@H1_1 # 2阶导公式中1阶导初始值的参数矩阵
# H2_2 = First # 2阶导公式中2阶导初始值的参数矩阵
#
# G3 = Diff@G2 # 2阶导公式中x的参数矩阵
# H3_1 = Diff@H2_1 # 3阶导公式中1阶导初始值的参数矩阵
# H3_2 = Diff@H2_2 # 3阶导公式中2阶导初始值的参数矩阵
#
# weight_comf = 0.1
# weight_jerk_s = 1
# weight_jerk_l = 10
#
# weight_eff = 2
# weight_eff_s = 1
# weight_eff_s_d = 0.5
#
# weight_safe = 2
# weight_safe_lateral_offset = 1


