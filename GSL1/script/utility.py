import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
import pandas as pd
import torch.nn.functional as F
import time
from iapws import IAPWS97
import CoolProp.CoolProp as CP

'''
class Standardizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, data):

        # 如果输入是多维张量，按时间维度计算均值和标准差

        # 如果是1维张量，直接计算
        mean = data.mean()
        std = data.std()

        # 保存均值和标准差
        self.means = mean
        self.stds = std

    def transform(self, data):

        mean = self.means
        std = self.stds
        return (data - mean) / std

    def inverse_transform(self, data):

        mean = self.means
        std = self.stds
        return data * std + mean
'''

class Standardizer:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        # 计算最小值和最大值
        self.min_val = data.min()
        self.max_val = data.max()

    def transform(self, data):
        # 执行最小-最大标准化
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data):
        # 反标准化
        return data * (self.max_val - self.min_val) + self.min_val


eos = 0
max_len = 80
use_eos = True

def find_stable_point(data, threshold=1e-5):
    seq_len = len(data)

    for i in range(seq_len - 1):
        # 计算相对误差
        rel_error = torch.abs((data[i + 1] - data[i]) / data[i].clamp(min=1e-8))
        if rel_error < threshold:
            return i

    return seq_len

def get_iter(df_list, len_train, len_val, args):
    list_train = []
    list_val = []
    list_test = []

    # 分别为训练集、验证集和测试集的稳定点创建列表
    stability_points_train = []
    stability_points_val = []
    stability_points_test = []

    # 用于存储每个样本、每个节点的标准化器
    standardizers = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for idx, cur_df in enumerate(df_list):
        data = torch.tensor(cur_df, dtype=torch.float32, device=device)
        # 获取数据的维度信息
        data_num, seq_len, node_num = data.shape
        stability_points = torch.zeros((data_num, node_num), dtype=torch.long, device=device)

        train_ = data[:len_train]
        val_ = data[len_train:len_train + len_val]  # 根据需要可调整验证集的长度
        test_ = data[len_train+len_val:]
        # 提前为每个 idx 和 node 创建标准化器
        for node in range(node_num):
            var_name = f"var_{idx}_node_{node}"
            standardizer = Standardizer()
            # 用于整个数据集的拟合，建议选择前70-80%的数据
            standardizer.fit(train_[:, :, node])
            standardizers[var_name] = standardizer
            for d in range(data_num):
                stable_point_index = find_stable_point(data[d, :, node])
                stability_points[d, node] = stable_point_index
                

        stability_points_train.append(stability_points[:len_train].unsqueeze(0))
        stability_points_val.append(stability_points[len_train:len_train + len_val].unsqueeze(0))
        stability_points_test.append(stability_points[len_train + len_val:].unsqueeze(0))

        for node in range(node_num):
            var_name = f"var_{idx}_node_{node}"
            train_[:, :, node] = standardizers[var_name].transform(train_[:, :, node])
            val_[:, :, node] = standardizers[var_name].transform(val_[:, :, node])
            test_[:, :, node] = standardizers[var_name].transform(test_[:, :, node])
            for d in range(len_train):
                stable_point_train = stability_points[d, node].item()
                if stable_point_train == 0:
                    stable_point_train += 1
                if args.use_eos==True:
                    train_[d, stable_point_train:, node] = args.eos
                else:
                    if stable_point_train < train_.shape[1]:
                        train_[d, stable_point_train:, node] = train_[d, stable_point_train, node].item()
            for d in range(len_train, len_train+len_val):
                stable_point_val = stability_points[d, node].item()
                if stable_point_val == 0:
                    stable_point_val += 1
                if args.use_eos==True:
                    val_[d-len_train, stable_point_val:, node] = args.eos
                else:
                    if stable_point_val < val_.shape[1]:
                        val_[d-len_train, stable_point_val:, node] = val_[d-len_train, stable_point_val, node].item()
            for d in range(len_val+len_train, data_num):
                stable_point_test = stability_points[d, node].item()
                if stable_point_test == 0:
                    stable_point_test += 1
                if args.use_eos==True:
                     test_[d-len_val-len_train, stable_point_test:, node] = args.eos
                else:
                    if stable_point_test < test_.shape[1]:
                        test_[d-len_val-len_train, stable_point_test:, node] = test_[d-len_val-len_train, stable_point_test, node].item()
       
        # 裁剪数据到 max_len
        train_ = train_[:, :args.max_len, :].unsqueeze(1)
        val_ = val_[:, :args.max_len, :].unsqueeze(1)
        test_ = test_[:, :args.max_len, :].unsqueeze(1)
        # 将处理后的数据添加到对应的列表
        list_train.append(train_)
        list_val.append(val_)
        list_test.append(test_)

    # 拼接不同特征的标准化数据
    data_train = torch.cat(list_train, dim=1)  # [train_size, feature, seq_len, node_num]
    data_val = torch.cat(list_val, dim=1)  # [val_size, feature, seq_len, node_num]
    data_test = torch.cat(list_test, dim=1)  # [test_size, feature, seq_len, node_num]


    # 将稳定点数据重新拼接
    data_train_point = torch.cat(stability_points_train, dim=0).permute(1, 0, 2)
    data_val_point = torch.cat(stability_points_val, dim=0).permute(1, 0, 2)
    data_test_point = torch.cat(stability_points_test, dim=0).permute(1, 0, 2)

    # 返回训练集、验证集、测试集，标准化实例，以及稳定点
    return data_train, data_val, data_test, standardizers, data_train_point, data_val_point, data_test_point


def evaluate_model(model, loss, data_iter_list, args, logger):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_steam, batch_air, batch_ele, batch_c1, batch_c2 in zip(*data_iter_list):
            xs = batch_steam.x
            xa = batch_air.x
            xe = batch_ele.x
            xc1 = batch_c1.x
            xc2 = batch_c2.x
            xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred = model(xs, xa, xe, xc1, xc2, use_teacher_forcing=True)
            l = loss(xs_pred[:, :, 1:, 1:], xs[:, :, 1:, 1:]) + loss(xa_pred[:, :, 1:, 1:], xa[:, :, 1:, 1:]) + loss(
                xe_pred[:, :, 1:, :], xe[:, :, 1:, :]) + loss(xc1_pred[:, :, 1:, :], xc1[:, :, 1:, :]) + loss(
                xc2_pred[:, :, 1:, :], xc2[:, :, 1:, :])

            l_sum += l.item() * xs.shape[0]
            n += xs.shape[0]
        mse = l_sum / n
        #logger.info(f"代码执行耗时: {sum_:.6f} 秒")
        logger.info(f'Test loss {mse:.6f}')

def calculate_mape(true, pred):
    # 避免除以零，添加一个小的epsilon
    epsilon = 1e-10
    mape = torch.abs((true - pred) / (true + epsilon))  # 计算绝对百分比误差
    return torch.mean(mape)  # 对所有元素求平均

def evaluate_sub(xs, xa, xe, xc1, xc2, xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred, score_list):

    xs = inverse_standardize(xs, score_list[0])
    xa = inverse_standardize(xa, score_list[1])
    xe = inverse_standardize(xe, score_list[2])
    xc1 = inverse_standardize(xc1, score_list[3])
    xc2 = inverse_standardize(xc2, score_list[4])

    xs_pred = inverse_standardize(xs_pred, score_list[0])
    xa_pred = inverse_standardize(xa_pred, score_list[1])
    xe_pred = inverse_standardize(xe_pred, score_list[2])
    xc1_pred = inverse_standardize(xc1_pred, score_list[3])
    xc2_pred = inverse_standardize(xc2_pred, score_list[4])

    mape_s = calculate_mape(xs, xs_pred)
    mape_a = calculate_mape(xa, xa_pred)
    mape_e = calculate_mape(xe, xe_pred)
    mape_c1 = calculate_mape(xc1, xc1_pred)
    mape_c2 = calculate_mape(xc2, xc2_pred)
    return mape_s, mape_a, mape_e, mape_c1, mape_c2


def evaluate_metric(model, loss, data_iter, score_list, args, logger):
    model.eval()
    with torch.no_grad():
        total_l, total_l_pinn, total_s, total_e, total_a, total_c1, total_c2 = 0, 0, 0, 0, 0, 0, 0
        sample_num = 0
        for batch_steam, batch_air, batch_ele, batch_c1, batch_c2 in zip(*data_iter):
            xs = batch_steam.x
            xa = batch_air.x
            xe = batch_ele.x
            xc1 = batch_c1.x
            xc2 = batch_c2.x
            point_s = batch_steam.y
            point_a = batch_air.y
            point_e = batch_ele.y
            point_c1 = batch_c1.y
            point_c2 = batch_c2.y
            xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred = model(xs, xa, xe, xc1, xc2, use_teacher_forcing=False)
            if args.use_eos:
                xs_pred = apply_stability_mask_eos(xs_pred, point_s)
                xa_pred = apply_stability_mask_eos(xa_pred, point_a)
                xe_pred = apply_stability_mask_eos(xe_pred, point_e)
                xc1_pred = apply_stability_mask_eos(xc1_pred, point_c1)
                xc2_pred = apply_stability_mask_eos(xc2_pred, point_c2)

            l = loss(xs_pred[:, :, 1:, 1:], xs[:, :, 1:, 1:]) + loss(xa_pred[:, :, 1:, 1:], xa[:, :, 1:, 1:]) + loss(
                xe_pred[:, :, 1:, :], xe[:, :, 1:, :]) + loss(xc1_pred[:, :, 1:, :], xc1[:, :, 1:, :]) + loss(
                xc2_pred[:, :, 1:, :], xc2[:, :, 1:, :])
            total_l += l

            if args.pinn:
                l_pinn = caculate_with_func(args, xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred, score_list)
                total_l_pinn += l_pinn.item() / args.factor * xs.shape[0]

            mape_s, mape_a, mape_e, mape_c1, mape_c2 = evaluate_sub(xs, xa, xe, xc1, xc2, xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred, score_list)

            total_s += mape_s * xs.shape[0]
            total_e += mape_e * xs.shape[0]
            total_a += mape_a * xs.shape[0]
            total_c1 += mape_c1 * xs.shape[0]
            total_c2 += mape_c2 * xs.shape[0]

            sample_num += xs.shape[0]

        logger.info(
            f'Loss {total_l / sample_num:.6f} \n'
            f'steam_MAPE {total_s / sample_num:.6f} | electricity_MAPE {total_e / sample_num:.6f} | compressed_air_MAPE {total_a / sample_num:.6f} | generator_MAPE {total_c1 / sample_num:.6f} | compressor_MAPE {total_c2 / sample_num:.6f}')
        if args.pinn:
            logger.info(f'| PINN平均相对误差 {total_l_pinn / sample_num:.6f}')


def custom_collate(data_list):
    x = torch.stack([data.x for data in data_list]) # 堆叠x
    y = torch.stack([data.y for data in data_list])
    batch_data = Batch()
    batch_data.x = x
    batch_data.y = y
    return batch_data


def apply_stability_mask(data, stability_points):
    batchsize, feature, seqlen, nodenum = data.shape

    # 遍历每个 batch、每个特征、每个节点
    for b in range(batchsize):
        for f in range(feature):
            for n in range(nodenum):
                # 获取该节点的稳定点索引
                stable_point = stability_points[b, f, n].item()  # 稳定点的时间步索引

                if stable_point < seqlen:  # 确保稳定点在序列长度内
                    # 将稳定点之后的数据替换为该稳定点前最后的数值
                    last_value = data[b, f, stable_point - 1, n]
                    data[b, f, stable_point:, n] = last_value
    return data

def apply_stability_mask_eos(data, stability_points):
    batchsize, feature, seqlen, nodenum = data.shape
    data_copy = data.clone()
    # 遍历每个 batch、每个特征、每个节点
    for b in range(batchsize):
        for f in range(feature):
            for n in range(nodenum):
                # 获取该节点的稳定点索引
                stable_point = stability_points[b, f, n].item()  # 稳定点的时间步索引

                if stable_point < seqlen:  # 确保稳定点在序列长度内
                    # 将稳定点之后的数据替换为该稳定点前最后的数值
                    data_copy[b, f, stable_point:, n] = 0
    return data_copy
'''
def replace_after_999(data, threshold=0):
    batchsize, feature, seqlen, nodenum = data.shape

    # 遍历每个 batch、每个特征、每个节点
    for b in range(batchsize):
        for f in range(feature):
            for n in range(nodenum):
                # 找到第一次出现999的位置
                index_999 = (data[b, f, :, n] == threshold).nonzero(as_tuple=False)

                if index_999.numel() > 0:  # 检查是否存在0
                    first_999_index = index_999[0].item()  # 获取第一次出现0的位置索引

                    if first_999_index > 0:
                        # 获取0之前的最后一个数值
                        last_value = data[b, f, first_999_index - 1, n]

                        # 将该位置及之后的所有数替换为该值
                        data[b, f, first_999_index:, n] = last_value
    return data

def replace_after_stable_point(data, tolerance=1e-6):
    batchsize, feature, seqlen, nodenum = data.shape

    for b in range(batchsize):
        for f in range(feature):
            for n in range(nodenum):
                for t in range(1, seqlen):
                    # 计算相对误差
                    relative_error = abs(data[b, f, t, n] - data[b, f, t - 1, n]) / abs(data[b, f, t - 1, n])

                    if relative_error < tolerance:
                        # 找到稳定点，替换后面的所有值
                        stable_value = data[b, f, t, n]
                        data[b, f, t:, n] = stable_value
                        break  # 找到一个稳定点后跳出循环
    return data
'''
def inverse_standardize(data, standardizers):

    batchsize, feature, seqlen, nodenum = data.shape

    # 遍历每个 batch, feature 和 node
    for b in range(batchsize):
        for f in range(feature):
            for n in range(nodenum):
                var_name = f"var_{f}_node_{n}"  # 生成唯一的 var_name

                # 从字典中获取对应的标准化器
                if var_name in standardizers:
                    standardizer = standardizers[var_name]
                    data[b, f, :, n] = standardizer.inverse_transform(data[b, f, :, n])
                else:
                    raise KeyError(f"Standardizer for {var_name} not found in standardizers")

    return data

def selected_pipe():
    steam1 = [59, 273, 118, 99, 120, 29, 110, 42, 282, 53, 23, 20, 28, 23, 151, 314, 114, 90, 19, 10, 44, 14, 102]
    steam2 = [0.6, 0.35, 0.25, 0.3, 0.6, 0.35, 0.6, 0.6, 0.45, 0.25, 0.45, 0.35, 0.45, 0.35, 0.6, 0.45, 0.4, 0.45,
              0.35, 0.45, 0.3, 0.45, 0.35]
    air1 = [1340, 2919, 2834, 2300, 1350, 1290]
    air2 = [0.8, 0.8, 0.6, 0.4, 0.4, 0.4]
    ele1 = [0, 0, 0, 0.1369, 0.23273, 0.43808, 0.53391, 0.116365, 0.016291]
    ele2 = [0.788544, 0.855625, 0.800865, 1.16365, 1.25948, 2.20409, 2.3273, 0.98568, 1.379952]
    ele3 = [0, 0, 0, 0.00642805, 0.00057706, 0.01117604, 0.01307524, 0.00544193, 0.00763331]
    edge_index = [[0, 1, 2, 2, 1, 5, 5, 7, 8, 9, 9, 11, 11, 13, 7, 15, 16, 15, 18, 18, 20, 16, 22, 27, 28, 29, 30, 30, 33, 35, 33, 35, 37, 38, 39, 40, 40, 39, 0, 0, 0, 24, 25, 26, 36],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30, 33, 35, 31, 32, 31, 32, 34, 34, 38, 39, 40, 41, 42, 43, 25, 26, 36, 27, 28, 29, 37]]
    return steam1, steam2, air1, air2, ele1, ele2, ele3, edge_index

def caculate_with_func(args, xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred, score_list):

    xs_pred = inverse_standardize(xs_pred, score_list[0])
    xa_pred = inverse_standardize(xa_pred, score_list[1])
    xe_pred = inverse_standardize(xe_pred, score_list[2])
    xc1_pred = inverse_standardize(xc1_pred, score_list[3])
    xc2_pred = inverse_standardize(xc2_pred, score_list[4])
    loss1 = steam_loss(xs_pred)
    loss2 = air_loss(xa_pred)
    loss3 = turbine_loss(xs_pred, xc1_pred, xc2_pred)
    return loss1 + loss2 + loss3

def steam_loss(xs):
    #[8,3,32,24]
    G, P, T = torch.split(xs, split_size_or_sections=1, dim=1)
    G = G.squeeze()
    P = P.squeeze()
    T = T.squeeze()

    batchsize, seqlen, _ = G.shape
    loss = 0
    for idx in range(batchsize):
        for step in range(seqlen):
            loss += func1(G[idx, step, :], P[idx, step, :], T[idx, step, :])
    return loss / batchsize / seqlen

def func1(G, P, T):
    edge_index = [
        [0, 1, 2, 2, 1, 5, 5, 7, 8, 9, 9, 11, 11, 13, 7, 15, 16, 15, 18, 18, 20, 16, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    steam1 = [59, 273, 118, 99, 120, 29, 110, 42, 282, 53, 23, 20, 28, 23, 151, 314, 114, 90, 19, 10, 44, 14, 102]
    steam2 = [0.6, 0.35, 0.25, 0.3, 0.6, 0.35, 0.6, 0.6, 0.45, 0.25, 0.45, 0.35, 0.45, 0.35, 0.6, 0.45, 0.4, 0.45,
              0.35, 0.45, 0.3, 0.45, 0.35]
    len_ = len(edge_index[0])
    sum = 0
    for i in range(len_):
        innode = edge_index[0][i]
        outnode = edge_index[1][i]
        l = steam1[i]
        d = steam2[i]
        f = judge_f(d)
        Tout = T[outnode].cpu().detach().numpy()
        Pout = P[outnode].cpu().detach().numpy() / 1E6

        vapor = IAPWS97(T=Tout.item(), P=Pout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        v = torch.tensor(vapor.v, device=device)
        q = 4.7819 * 1E-5 * (293.15 / 101325) * (
                    ((P[outnode]) ** 2 - (P[innode]) ** 2) / (f * l / 1000 * 0.62069 * (T[innode] + T[outnode]) / 2)) ** 0.5 * (
                        d * 1000) ** 2.5
        G_cal = q / 3600 / v
        sum += (G_cal - G[outnode]) / G[outnode]
        #sum_percent += (G_cal - G[outnode]) / G[outnode]
    return sum / len_ #, sum_percent / len_

def judge_f(d):
    if d <= 0.3:
        return 0.013
    elif d > 0.3 and d <=0.45:
        return 0.013 - (d-0.3)/(0.45-0.3)*0.001
    elif d > 0.45 and d <= 0.65:
        return 0.012 - (d-0.45)/(0.65-0.45)*0.001
    elif d > 0.65 and d <= 1.05:
        return 0.011 - (d-0.65)/(1.05-0.65)*0.001
    elif d > 1.05 and d <= 2.025:
        return 0.01 - (d-1.05)/(2.025-1.05)*0.001
    elif d > 2.025 and d <= 4.06:
        return 0.09 - (d-2.025)/(4.06-2.025)*0.001
    else:
        return 0.0078

def air_loss(xa):
    # [8,3,32,24]
    G, P, = torch.split(xa, split_size_or_sections=1, dim=1)
    G = G.squeeze()
    P = P.squeeze()
    batchsize, seqlen, _ = G.shape
    loss = 0
    for idx in range(batchsize):
        for step in range(seqlen):
            loss += func2(G[idx, step, :], P[idx, step, :])
    return loss / batchsize / seqlen

def func2(G, P):
    edge_index = [
        [0, 1, 2, 3, 3, 2],
        [1, 2, 3, 4, 5, 6]]
    air1 = [1340, 2919, 2834, 2300, 1350, 1290]
    air2 = [0.8, 0.8, 0.6, 0.4, 0.4, 0.4]
    len_ = len(edge_index[0])
    sum = 0
    for i in range(len_):
        innode = edge_index[0][i]
        outnode = edge_index[1][i]
        l = air1[i]
        d = air2[i]
        f = judge_f(d)
        Pout = P[outnode].cpu().detach().numpy()

        temperature = 387  # 单位：开尔文(K)
        pressure = Pout.item()  # 单位：帕斯卡(Pa)
        v = CP.PropsSI('V', 'T', temperature, 'P', pressure, 'Air')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        v = torch.tensor(v, device=device)
        q = 4.7819 * 1E-5 * (293.15 / 101325) * (
                    ((P[outnode]) ** 2 - (P[innode]) ** 2) / (f * l / 1000 * 0.62069 * 387) ** 0.5) * (
                        d * 1000) ** 2.5
        G_cal = q / 3600 / v
        sum += (G_cal - G[outnode]) / G[outnode]
        #sum_percent += (G_cal - G[outnode]) / G[outnode]
    return sum / len_ #,sum_percent / len_

def turbine_loss(xs, xc1, xc2):
    G, P, T = torch.split(xs, split_size_or_sections=1, dim=1)
    G = G.squeeze()
    P = P.squeeze()
    T = T.squeeze()
    active, _, = torch.split(xc1, split_size_or_sections=1, dim=1)
    active = active.squeeze()
    power = xc2.squeeze()

    batchsize, seqlen, _ = G.shape
    loss = 0
    for idx in range(batchsize):
        for step in range(seqlen):
            loss += func3(G[idx, step, 0], P[idx, step, 0], T[idx, step, 0], active[idx, step, 1:], power[idx, step])    #G*(h1-h2) = ξ1*active1 + ξ2*active2 + ξ3*power
    return loss / batchsize / seqlen

def func3(G, P, T, active, power):
    Tin = T.item()
    Pin = P.item() / 1E6
    Tout = 273.15 + 20
    Pout = 0.101325

    vapor_in = IAPWS97(T=Tin, P=Pin)
    vapor_out = IAPWS97(T=Tout, P=Pout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h1 = torch.tensor(vapor_in.v, device=device)
    h2 = torch.tensor(vapor_out.v, device=device)

    sum = (G.item() * (h1 - h2) / 1000 - 0.99 * power - 0.96 * active[0] -0.96 * active[1]) / (G.item() * (h1 - h2) / 1000)
    return sum
