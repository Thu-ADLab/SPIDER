import numpy as np

def test1(xs, vxs, axs, xe, vxe, axe, te):
    a0 = xs
    a1 = vxs
    a2 = axs / 2.0

    A = np.array([[te ** 3, te ** 4, te ** 5],
                  [3 * te ** 2, 4 * te ** 3, 5 * te ** 4],
                  [6 * te, 12 * te ** 2, 20 * te ** 3]])
    b = np.array([xe - a0 - a1 * te - a2 * te ** 2,
                  vxe - a1 - 2 * a2 * te,
                  axe - 2 * a2])
    x = np.linalg.solve(A, b)

    a3 = x[0]
    a4 = x[1]
    a5 = x[2]
    t_range = te

def test2(xs, vxs, axs, xe, vxe, axe,ts, te):


    A = np.array([
        [ts**5, ts**4,ts**3,ts**2,ts, 1],
        [5 * ts ** 4, 4 * ts ** 3, 3 * ts ** 2, 2*ts, 1, 0],
        [20 * ts ** 3, 12 * ts ** 2, 6 * ts, 2,0,0],
        [te ** 5, te ** 4, te ** 3, te ** 2, te, 1],
        [5 * te ** 4, 4 * te ** 3, 3 * te ** 2, 2 * te, 1, 0],
        [20 * te ** 3, 12 * te ** 2, 6 * te, 2,0,0]
    ])

    b = np.array([
        xs,
        vxs,
        axs,
        xe,
        vxe,
        axe
    ])
    x = np.linalg.solve(A, b)
    coef = x
    t_range = te


if __name__ == '__main__':
    from time import time

    xs, vxs, axs, xe, vxe, axe, ts, te = 3.5, 0, 0, 0, 0, 0, 0, 2

    t = 0.
    for _ in range(10000):
        t1 = time()
        test1(xs, vxs, axs, xe, vxe, axe, te)
        t += time() - t1
    print(t)

    t = 0.
    for _ in range(10000):
        t1 = time()
        test2(xs, vxs, axs, xe, vxe, axe,ts, te)
        t += time() - t1
    print(t)
