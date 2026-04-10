import numpy as np


def trim_segment(arr, index, num_area, AAP, ADP, ASP):
    """
    trimming segment from polar backscatter by index
    :param arr: input polar backscatter (theta, range)
    :param index: index of segment angle
    :return:
    """
    win = int(1 / num_area * AAP)
    lb_t = int(index * win - ASP)
    rb_t = int(index * win + ASP)

    lb_r = ADP - ASP
    rb_r = ADP + ASP

    if lb_t < 0:
        return np.concatenate((arr[lb_t:, lb_r:rb_r], arr[:int(2 * ASP + lb_t), lb_r:rb_r]), axis=0)
    elif rb_t > AAP:
        return np.concatenate((arr[AAP - rb_t:, lb_r:rb_r], arr[:int(2 * ASP - (rb_t - AAP)), lb_r:rb_r]), axis=0)
    else:
        return arr[lb_t:rb_t, lb_r:rb_r]


def calc_dir_ind(arr, num, AAP, ADP, ASP):
    """
    calculate main wave direction
    """
    rose = np.array([
        np.std(np.sum(trim_segment(arr, n, num, AAP, ADP, ASP), axis=0), axis=0)
        for n in range(num)
    ])
    return int(np.argmax(rose))
