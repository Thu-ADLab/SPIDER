# mymodule.pyx

# 引入 Cython 的声明
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef evaluate(double[::1] x, double[::1] coef, double valid_x_lower, double valid_x_upper, int order, 
    double y_0, double y_prime_0, double y_end, double y_prime_end):
    """
    Evaluate the polynomial function for each element in x within the valid range.
    
    Parameters:
    - x: Input array of values.
    - coef: Coefficients of the polynomial.
    - valid_x_lower: Lower bound of the valid range for x values.
    - valid_x_upper: Upper bound of the valid range for x values.
    - order: 这里order的意思是，coef对应着原来多项式的几阶导数，用来在外推的时候确定外推规则
    
    Returns:
    - Output array of evaluated values.
    """
    # 获取数组大小
    cdef Py_ssize_t n = x.shape[0]

    # 创建输出数组
    cdef double[::1] y = cython.view.array(shape=(n,), itemsize=sizeof(double), format="d")

    cdef double term
    cdef int j

    # 遍历输入数组
    for i in range(n):
        # 检查 x 是否在有效范围内
        if valid_x_lower <= x[i] <= valid_x_upper:
            # 计算多项式函数值
            y[i] = 0.0
            term = 1.0
            for j in range(len(coef)):
                y[i] += coef[len(coef)-j-1] * term
                term *= x[i]
        else:
            # 超出有效范围，设置为 0
            y[i] = extrapolate(x[i], valid_x_lower, valid_x_upper, order,
                            y_0, y_prime_0, y_end, y_prime_end)

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double evaluate_scalar(double x, double[::1] coef, double valid_x_lower, double valid_x_upper, int order,
    double y_0, double y_prime_0, double y_end, double y_prime_end):
    """
    Evaluate the polynomial function for a single scalar value x within the valid range.

    Parameters:
    - x: Input scalar value.
    - coef: Coefficients of the polynomial.
    - valid_x_lower: Lower bound of the valid range for x values.
    - valid_x_upper: Upper bound of the valid range for x values.
    - order: Order of the polynomial.
    - y_0: Initial value.
    - y_prime_0: Initial derivative value.
    - y_end: Final value.
    - y_prime_end: Final derivative value.

    Returns:
    - Output scalar of evaluated value.
    """
    cdef double y
    cdef double term
    cdef int j

    # 检查 x 是否在有效范围内
    if valid_x_lower <= x <= valid_x_upper:
        # 计算多项式函数值
        y = 0.0
        term = 1.0
        for j in range(len(coef)):
            y += coef[len(coef)-j-1] * term
            term *= x
        return y
    else:
        # 超出有效范围，设置为 0
        return extrapolate(x, valid_x_lower, valid_x_upper, order,
                           y_0, y_prime_0, y_end, y_prime_end)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef extrapolate(double x, double valid_x_lower, double valid_x_upper, int order,
    double y_0, double y_prime_0, double y_end, double y_prime_end):
    '''
    在超出已知数据范围的点上进行插值/外推，默认保持二阶导为0然后外推，即保持起点或终点的斜率
    注意：如果 x 中的任何点在有效范围内，则相应的 y 维持为 0 值
    '''
    cdef double val


    if order == 0:

        if x > valid_x_upper:
            val = y_end + y_prime_end * (x - valid_x_upper)
        elif x < valid_x_lower:
            val = y_0 + y_prime_0 * (x - valid_x_lower)
        else:
            val = 0.0

    elif order == 1:

        if x > valid_x_upper:
            val = y_prime_end
        elif x < valid_x_lower:
            val = y_prime_0
        else:
            val = 0.0
    else:
        val = 0.0  # 因为更高阶导数无论如何都是 0

    return val

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef extrapolate(double x, double[::1] coef, double valid_x_lower, double valid_x_upper, int order):
#    '''
#    在超出已知数据范围的点上进行插值/外推，默认保持二阶导为0然后外推，即保持起点或终点的斜率
#    注意：如果 x 中的任何点在有效范围内，则相应的 y 维持为 0 值
#    '''
#    cdef double val
#    cdef double y_prime_0
#    cdef double y_prime_end
#    cdef double y_0
#    cdef double y_end
#    cdef double[::1] der_coef
#
#
#    if order == 0:
#        der_coef = compute_derivative_coefs(coef, 1) # der_coef在被反复计算！
#
#        if x > valid_x_upper:
#            # y_prime_end在被反复计算！
#            y_prime_end = evaluate(np.array([valid_x_upper]), der_coef, valid_x_lower, valid_x_upper, order+1)[0]
#            y_end = evaluate(np.array([valid_x_upper]),coef, valid_x_lower, valid_x_upper, order)[0]
#            val = y_end + y_prime_end * (x - valid_x_upper)
#        elif x < valid_x_lower:
#            y_prime_0 = evaluate(np.array([valid_x_lower]), der_coef, valid_x_lower, valid_x_upper,  order+1)[0]
#            y_0 = evaluate(np.array([valid_x_lower]),coef, valid_x_lower, valid_x_upper, order)[0]
#            val = y_0 + y_prime_0 * (x - valid_x_lower)
#        else:
#            val = 0.0
#
#    elif order == 1:
#        der_coef = compute_derivative_coefs(coef, 1)
#
#        if x > valid_x_upper:
#            y_prime_end = evaluate(np.array([valid_x_upper]), der_coef, valid_x_lower, valid_x_upper, order+1)[0]
#            val = y_prime_end
#        elif x < valid_x_lower:
#            y_prime_0 = evaluate(np.array([valid_x_lower]), der_coef, valid_x_lower, valid_x_upper, order+1)[0]
#            val = y_prime_0
#        else:
#            val = 0.0
#    else:
#        val = 0.0  # 因为更高阶导数无论如何都是 0
#
#    return val
#
#
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef compute_derivative_coefs(double[::1] coef, int derivative_order):
#    """
#    计算多项式函数的导数的参数。
#
#    Parameters:
#    - coef: 多项式系数
#    - derivative_order: 导数的阶数
#
#    Returns:
#    - 导数的系数
#    """
#    cdef Py_ssize_t n = coef.shape[0]
#
#    if derivative_order >= n:
#        # 如果导数的阶数大于等于系数的阶数，导数的系数全为零
#        return np.zeros(n, dtype=np.float64)
#
#    # 计算导数的系数
#    cdef double[::1] derivative_coefs = np.zeros(n - derivative_order, dtype=np.float64)
#    for i in range(n - derivative_order):
#        derivative_coefs[i] = coef[i + derivative_order] * (i + derivative_order)
#    return derivative_coefs
#