import numpy as np

def perform_first_order_check(jlist, j0, gradj0, ds, epslist):
    # j0: function value at x0
    # gradj0: gradient value at x0
    # epslist: list of decreasing eps-values
    # jlist: list of function values at x0+eps*ds for all eps in epslist
    diff0 = []
    diff1 = []
    order0 = []
    order1 = []
    i = 0
    for eps in epslist:
        je = jlist[i]
        di0 = je - j0
        di1 = je - j0 - eps * np.dot(gradj0, ds)
        diff0.append(abs(di0))
        diff1.append(abs(di1))
        if i == 0:
            order0.append(0.0)
            order1.append(0.0)
        if i > 0:
            order0.append(np.log(diff0[i - 1] / diff0[i]) / np.log(epslist[i - 1] / epslist[i]))
            order1.append(np.log(diff1[i - 1] / diff1[i]) / np.log(epslist[i - 1] / epslist[i]))
        i = i + 1
    for i in range(len(epslist)):
        print('eps\t', epslist[i], '\t\t check continuity\t', order0[i], '\t\t diff0 \t', diff0[i],
              '\t\t check derivative \t', order1[i], '\t\t diff1 \t', diff1[i], '\n'),

    return order1[-1], diff1[-1]