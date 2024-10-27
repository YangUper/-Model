import random

import numpy as np

# 该函数的整体作用是：对输入的四维数组 data_numpy（形状为 (C, T, V, M)）进行时间维度的下采样。
# 通过 step 参数确定下采样的步长，通过 random_sample 参数决定是否随机选择开始帧。
def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]

# 对输入的四维数组 data_numpy（形状为 (C, T, V, M)）进行空间维度的下采样。
# 它将时间维度和人数维度部分结合，结果是一个新的四维数组，时间维度的长度缩短为 T / step，而人数维度扩展为 step * M。
def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)

#函数首先检查 mean 是否为 0，如果是，直接返回。
# 计算输入数据中的有效帧，即判断在时间维度上哪些帧包含非零值（认为包含信息的帧）。
# 找出第一个有效帧和最后一个有效帧的索引。
# 对这些有效帧的数据，减去均值 mean，实现均值归一化或平移。
# 返回处理后的数据。
def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy

#该函数主要用于对时间序列数据进行处理，特别是在深度学习中的视频数据或人体动作序列中，
# 输入的时间帧数往往不一致。通过填充，使所有输入的数据都具有相同的时间帧数，这样可以统一输入形状，便于后续处理和训练。
def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy

#该函数常用于处理时间序列数据，特别是在深度学习中的视频、动作序列或人体关键点数据中。
# 当输入数据的长度不一致时，通过随机选择或自动填充，保证所有输入数据的长度统一为 size，这样便于模型在批量处理中使用。
def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

#这段代码定义了一个名为 random_move 的函数，用于对输入的四维数据 data_numpy 进行随机的旋转、缩放和平移操作。
# 该函数主要用于增强数据，使其具有更多的随机性和多样性，特别适用于处理动作序列数据（如人体骨骼数据），
# 通过这些随机变换来模拟实际中的多样性。
def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy

#这段代码定义了一个名为 random_shift 的函数，用于对输入的四维数据 data_numpy 进行随机的时间段偏移操作。
# 其主要目的是将有效的时间段数据（非零帧）随机移动到新的位置，常用于数据增强。
def random_shift(data_numpy):
    # input: C,T,V,M 偏移其中一段
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
