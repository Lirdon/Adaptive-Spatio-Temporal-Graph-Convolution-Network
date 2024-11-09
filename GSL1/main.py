import logging
import argparse
import random
import os
import tqdm
import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from model import my_model
from script import get_data, utility, earlystopping
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from script.logger_factory import *
from script.utility import caculate_with_func, get_iter
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from datetime import datetime

logger = setup_logger("main")


# logger = setup_logger("test")
# logger = setup_logger("hyper_search")


def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='RTGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--need_grad', type=bool, default=True, help='default as True')
    parser.add_argument('--pinn', type=bool, default=False, help='only be True when searching hyperparameters')
    parser.add_argument('--use_gsl', type=bool, default=True)
    parser.add_argument('--use_eos', type=bool, default=False)
    parser.add_argument("--time_model", type=str, default="attention", help="attention, RNN, CVAE")
    parser.add_argument('--eos', type=int, default=0)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--n_vertex', type=int, default=44)
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument("--graph_skip_conn", type=float, default=0.7796910002727695, help="Default is 10.")
    parser.add_argument("--graph_include_self", type=bool, default=True, help="Default is 10.")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Default is 16.")
    parser.add_argument("--graph_type", type=str, default="epsilonNN", help="epsilonNN, KNN, prob")
    parser.add_argument("--phy_top_type", type=str, default="KNN", help="epsilon, KNN")
    parser.add_argument("--graph_metric_type", type=str, default="gat_attention",
                        help="gat_attention, attention, weighted_cosine, kernel, transformer, cosine, mlp, multi_mlp")
    parser.add_argument("--num_per", type=int, default=3, help="Default is 16")
    parser.add_argument("--feature_denoise", type=bool, default=False, help="Default is True.")
    parser.add_argument("--top_k", type=int, default=6, help="Default is 10.")
    parser.add_argument("--epsilon", type=float, default=0.5968501579464871, help="Default is 10.")
    parser.add_argument('--steam_dim', type=int, default=3)
    parser.add_argument('--air_dim', type=int, default=2)
    parser.add_argument('--ele_dim', type=int, default=2)
    parser.add_argument('--c1_dim', type=int, default=2)
    parser.add_argument('--c2_dim', type=int, default=1)
    parser.add_argument('--graph_dim', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.2169234737081301)
    parser.add_argument('--lr', type=float, default=0.0001930783753654713, help='learning rate')
    parser.add_argument('--weight_type', type=int, default=3, help='1 for none, 2 for *, 3 for /')
    parser.add_argument('--weight_decay_rate', type=float, default=0.00020034427927560742,
                        help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    # parser.add_argument('--step_size', type=int, default=10)
    # parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--T_max', type=int, default=8)
    parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--lam', type=float, default=0.05)
    parser.add_argument('--Tb', type=float, default=288.15)
    parser.add_argument('--Pb', type=float, default=101325)
    parser.add_argument('--Sg', type=float, default=0.62069)

    args = parser.parse_args()
    # logger.info('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []  # +args.time_dim
    blocks.append([args.time_dim])  # 考虑增加特征维度后是并行计算
    for l in range(args.stblock_num):
        blocks.append([args.graph_dim, args.graph_dim, args.graph_dim])  # 考虑图部分变换维度，时间序列应该怎么变化以匹配

    return args, device, blocks


def data_preparate_steam(GATG, GATP, GATT):
    G_df, P_df, T_df = get_data.get_node_data_steam(GATG, GATP, GATT)
    data_col = G_df.shape[0]
    val_and_test_rate = args.valid_ratio
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    data_train, data_val, data_test, S_score, stability_points_train, stability_points_val, stability_points_test = get_iter(
        [G_df, P_df, T_df], len_train, len_val, args)
    data = {
        'data_train': data_train,
        'data_val': data_val,
        'data_test': data_test,
        'stability_points_train': stability_points_train,
        'stability_points_val': stability_points_val,
        'stability_points_test': stability_points_test,
    }
    if args.use_eos == True:
        with open('data/eos/data_steam.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/eos/score_steam.pkl', 'wb') as f:
            pickle.dump(S_score, f)
    else:
        with open('data/noeos/data_steam.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/noeos/score_steam.pkl', 'wb') as f:
            pickle.dump(S_score, f)


def data_preparate_c1(GATG, GATP):
    G_df, P_df = get_data.get_node_data_c1(GATG, GATP)

    data_col = G_df.shape[0]
    val_and_test_rate = args.valid_ratio
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    data_train, data_val, data_test, S_score, stability_points_train, stability_points_val, stability_points_test = get_iter(
        [G_df, P_df], len_train, len_val, args)
    data = {
        'data_train': data_train,
        'data_val': data_val,
        'data_test': data_test,
        'stability_points_train': stability_points_train,
        'stability_points_val': stability_points_val,
        'stability_points_test': stability_points_test,
    }
    if args.use_eos == True:
        with open('data/eos/data_c1.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/eos/score_c1.pkl', 'wb') as f:
            pickle.dump(S_score, f)
    else:
        with open('data/noeos/data_c1.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/noeos/score_c1.pkl', 'wb') as f:
            pickle.dump(S_score, f)


def data_preparate_c2(GATG):
    G_df = get_data.get_node_data_c2(GATG)

    data_col = G_df.shape[0]
    val_and_test_rate = args.valid_ratio
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    data_train, data_val, data_test, S_score, stability_points_train, stability_points_val, stability_points_test = get_iter(
        [G_df], len_train, len_val, args)
    data = {
        'data_train': data_train,
        'data_val': data_val,
        'data_test': data_test,
        'stability_points_train': stability_points_train,
        'stability_points_val': stability_points_val,
        'stability_points_test': stability_points_test,
    }
    if args.use_eos == True:
        with open('data/eos/data_c2.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/eos/score_c2.pkl', 'wb') as f:
            pickle.dump(S_score, f)
    else:
        with open('data/noeos/data_c2.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/noeos/score_c2.pkl', 'wb') as f:
            pickle.dump(S_score, f)


def data_preparate_air(GATG, GATP):
    G_df, P_df = get_data.get_node_data_air(GATG, GATP)

    data_col = G_df.shape[0]
    val_and_test_rate = args.valid_ratio
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    data_train, data_val, data_test, S_score, stability_points_train, stability_points_val, stability_points_test = get_iter(
        [G_df, P_df], len_train, len_val, args)
    data = {
        'data_train': data_train,
        'data_val': data_val,
        'data_test': data_test,
        'stability_points_train': stability_points_train,
        'stability_points_val': stability_points_val,
        'stability_points_test': stability_points_test,
    }
    if args.use_eos == True:
        with open('data/eos/data_air.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/eos/score_air.pkl', 'wb') as f:
            pickle.dump(S_score, f)
    else:
        with open('data/noeos/data_air.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/noeos/score_air.pkl', 'wb') as f:
            pickle.dump(S_score, f)


def data_preparate_ele(GATG, GATP):
    G_df, P_df = get_data.get_node_data_ele(GATG, GATP)
    data_col = G_df.shape[0]
    val_and_test_rate = args.valid_ratio
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)
    data_train, data_val, data_test, S_score, stability_points_train, stability_points_val, stability_points_test = get_iter(
        [G_df, P_df], len_train, len_val, args)
    data = {
        'data_train': data_train,
        'data_val': data_val,
        'data_test': data_test,
        'stability_points_train': stability_points_train,
        'stability_points_val': stability_points_val,
        'stability_points_test': stability_points_test,
    }

    if args.use_eos == True:
        with open('data/eos/data_ele.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/eos/score_ele.pkl', 'wb') as f:
            pickle.dump(S_score, f)
    else:
        with open('data/noeos/data_ele.pickle', 'wb') as f:
            pickle.dump(data, f)
        with open('data/noeos/score_ele.pkl', 'wb') as f:
            pickle.dump(S_score, f)


def load_data(args, data_paths, Sscore_paths):
    all_S_scores = []
    all_train_iters = []
    all_val_iters = []
    all_test_iters = []

    # 确保data_paths和Sscore_paths长度一致
    assert len(data_paths) == len(Sscore_paths), "data_paths和Sscore_paths长度不一致！"

    # 遍历所有文件
    for data_path, Sscore_path in zip(data_paths, Sscore_paths):
        # 加载data文件
        if args.use_eos:
            data_path = "data/eos/" + data_path
            Sscore_path = "data/eos/" + Sscore_path
        else:
            data_path = "data/eos/" + data_path
            Sscore_path = "data/eos/" + Sscore_path

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        data_train = data['data_train']
        data_val = data['data_val']
        data_test = data['data_test']
        stability_points_train = data['stability_points_train']
        stability_points_val = data['stability_points_val']
        stability_points_test = data['stability_points_test']

        # 加载Sscore文件
        with open(Sscore_path, 'rb') as f:
            S_score = pickle.load(f)

        # 重写每个批次数据集的构建方法
        data_train = get_data.data_split(data_train, stability_points_train)

        data_val = get_data.data_split(data_val, stability_points_val)
        data_test = get_data.data_split(data_test, stability_points_test)

        # 创建DataLoader迭代器
        train_iter = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=utility.custom_collate)
        val_iter = DataLoader(data_val, batch_size=args.batch_size, shuffle=True, collate_fn=utility.custom_collate)
        test_iter = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, collate_fn=utility.custom_collate)

        # 将S_score和DataLoader添加到结果列表中
        all_S_scores.append(S_score)
        all_train_iters.append(train_iter)
        all_val_iters.append(val_iter)
        all_test_iters.append(test_iter)

    return all_S_scores, all_train_iters, all_val_iters, all_test_iters


def prepare_model(args, blocks):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    model = my_model.STGCNGraphConv(args, blocks).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)

    return loss, es, model, optimizer, scheduler


def train(score_list, loss, args, optimizer, scheduler, es, model, train_iter_list, val_iter_list):
    best_loss = torch.inf
    for epoch in tqdm(range(args.epochs)):
        l_sum, l_pinn_sum, n = 0.0, 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for batch_steam, batch_air, batch_ele, batch_c1, batch_c2 in tqdm(zip(*train_iter_list)):
            xs = batch_steam.x
            xa = batch_air.x
            xe = batch_ele.x
            xc1 = batch_c1.x
            xc2 = batch_c2.x

            [xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred] = model(xs, xa, xe, xc1, xc2, use_teacher_forcing=True)

            if args.use_eos:
                point_s, point_a, point_e, point_c1, point_c2 = batch_steam.y, batch_air.y, batch_ele.y, batch_c1.y, batch_c2.y
                xs_pred = utility.apply_stability_mask_eos(xs_pred, point_s)
                xa_pred = utility.apply_stability_mask_eos(xa_pred, point_a)
                xe_pred = utility.apply_stability_mask_eos(xe_pred, point_e)
                xc1_pred = utility.apply_stability_mask_eos(xc1_pred, point_c1)
                xc2_pred = utility.apply_stability_mask_eos(xc2_pred, point_c2)

            l = loss(xs_pred[:, :, 1:, 1:], xs[:, :, 1:, 1:]) + loss(xa_pred[:, :, 1:, 1:], xa[:, :, 1:, 1:]) + loss(
                xe_pred[:, :, 1:, :], xe[:, :, 1:, :]) + loss(xc1_pred[:, :, 1:, :], xc1[:, :, 1:, :]) + loss(
                xc2_pred[:, :, 1:, :], xc2[:, :, 1:, :])

            # 在训练或验证计算损失时，应当保留序列全部信息；而在推理时，应当在截断后再计算MAPE

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * xs.shape[0]
            n += xs.shape[0]
        scheduler.step()

        if args.pinn:
            val_loss, val_pinn_loss = val(model, val_iter_list, loss, score_list)
            logger.info(
                'Epoch: {:03d} | Train loss: {:.6f} | Train pinn loss: {:.6f} | Val loss: {:.6f} | Val pinn loss: {:.6f} |'.format(
                    epoch + 1, l_sum / n, l_pinn_sum / n, val_loss, val_pinn_loss))
        else:
            val_loss = val(model, val_iter_list, loss, score_list)
            logger.info(
                'Epoch: {:03d} | Train loss: {:.6f} | Val loss: {:.6f} |'.format(epoch + 1, l_sum / n, val_loss))

        save_dir = f"checkpoint"
        os.makedirs(save_dir, exist_ok=True)

        if best_loss > val_loss:
            best_loss = val_loss
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"model_epoch_{current_time}_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at {save_path}")
        '''
        if (epoch + 1) % args.save_interval == 0:
            save_path = f"checkpoint/index_{args.selected_index}/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at {save_path}")
            #print(f"Model saved at {save_path}")
        '''

        if es.step(val_loss, logger):  # 加一个存储的判断
            logger.info('Early stopping.')
            # print('Early stopping.')
            break


def search_train(score_list, loss, args, optimizer, scheduler, es, model, train_iter_list, val_iter_list,
                 test_iter_list):
    logger.info('------------------------------')
    best_loss = torch.inf
    for epoch in tqdm(range(args.epochs)):
        l_sum, l_pinn_sum, n = 0.0, 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for batch_steam, batch_air, batch_ele, batch_c1, batch_c2 in zip(*train_iter_list):
            xs = batch_steam.x
            xa = batch_air.x
            xe = batch_ele.x
            xc1 = batch_c1.x
            xc2 = batch_c2.x

            [xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred] = model(xs, xa, xe, xc1, xc2, use_teacher_forcing=True)
            if args.use_eos:
                point_s, point_a, point_e, point_c1, point_c2 = batch_steam.y, batch_air.y, batch_ele.y, batch_c1.y, batch_c2.y
                xs_pred = utility.apply_stability_mask_eos(xs_pred, point_s)
                xa_pred = utility.apply_stability_mask_eos(xa_pred, point_a)
                xe_pred = utility.apply_stability_mask_eos(xe_pred, point_e)
                xc1_pred = utility.apply_stability_mask_eos(xc1_pred, point_c1)
                xc2_pred = utility.apply_stability_mask_eos(xc2_pred, point_c2)

            l = loss(xs_pred[:, :, 1:, 1:], xs[:, :, 1:, 1:]) + loss(xa_pred[:, :, 1:, 1:], xa[:, :, 1:, 1:]) + loss(
                xe_pred[:, :, 1:, :], xe[:, :, 1:, :]) + loss(xc1_pred[:, :, 1:, :], xc1[:, :, 1:, :]) + loss(
                xc2_pred[:, :, 1:, :], xc2[:, :, 1:, :])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * xs.shape[0]
            n += xs.shape[0]
            temp = l_sum / n

        scheduler.step()
        if args.pinn:
            val_loss, val_pinn_loss = val(model, val_iter_list, loss, score_list)
            logger.info(
                'Epoch: {:03d} | Train loss: {:.6f} | Val loss: {:.6f} | Val pinn loss: {:.6f} |'.format(
                    epoch + 1, l_sum / n, val_loss, val_pinn_loss))
        else:
            val_loss = val(model, val_iter_list, loss, score_list)
            logger.info(
                'Epoch: {:03d} | Train loss: {:.6f} | Val loss: {:.6f} |'.format(epoch + 1, l_sum / n, val_loss))

        if math.isnan(val_loss):
            val_loss = 100
            break
        save_dir = f"hyper_search/checkpoint"
        os.makedirs(save_dir, exist_ok=True)
        if best_loss > val_loss:
            best_loss = val_loss
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"model_epoch_{current_time}_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at {save_path}")
            test(score_list, loss, model, test_iter_list, args, logger)
        if es.step(val_loss, logger):  # 加一个存储的判断
            # logger.info('Early stopping.')
            # print('Early stopping.')
            break
    if args.pinn:
        return val_loss + val_pinn_loss  # 或者只要pinnloss
    else:
        return val_loss


@torch.no_grad()
def val(model, val_iter_list, loss, score_list):
    model.eval()
    l_sum, l_pinn_sum, n = 0.0, 0.0, 0
    for batch_steam, batch_air, batch_ele, batch_c1, batch_c2 in zip(*val_iter_list):
        xs = batch_steam.x
        xa = batch_air.x
        xe = batch_ele.x
        xc1 = batch_c1.x
        xc2 = batch_c2.x

        xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred = model(xs, xa, xe, xc1, xc2, use_teacher_forcing=False)
        if args.use_eos:
            point_s, point_a, point_e, point_c1, point_c2 = batch_steam.y, batch_air.y, batch_ele.y, batch_c1.y, batch_c2.y
            xs_pred = utility.apply_stability_mask_eos(xs_pred, point_s)
            xa_pred = utility.apply_stability_mask_eos(xa_pred, point_a)
            xe_pred = utility.apply_stability_mask_eos(xe_pred, point_e)
            xc1_pred = utility.apply_stability_mask_eos(xc1_pred, point_c1)
            xc2_pred = utility.apply_stability_mask_eos(xc2_pred, point_c2)
        if args.pinn:
            l_pinn = caculate_with_func(args, xs_pred, xc1_pred, xe_pred, xc2_pred, xa_pred, score_list)
            l_pinn_sum += l_pinn.item() / args.factor * xs.shape[0]

        l = loss(xs_pred[:, :, 1:, 1:], xs[:, :, 1:, 1:]) + loss(xa_pred[:, :, 1:, 1:], xa[:, :, 1:, 1:]) + loss(
            xe_pred[:, :, 1:, :], xe[:, :, 1:, :]) + loss(xc1_pred[:, :, 1:, :], xc1[:, :, 1:, :]) + loss(
            xc2_pred[:, :, 1:, :], xc2[:, :, 1:, :])
        l_sum += l.item() * xs.shape[0]
        n += xs.shape[0]
    if args.pinn:
        return l_sum / n, l_pinn_sum / n
    else:
        return l_sum / n


@torch.no_grad()
def test(score_list, loss, model, test_iter_list, args, logger):
    model.eval()
    # utility.evaluate_model(model, loss, test_iter_list, args, logger)   #把这个loss计算合并到下面metric计算里
    utility.evaluate_metric(model, loss, test_iter_list, score_list, args, logger)


search_space = [
    Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
    Real(1e-4, 1e-2, prior='log-uniform', name='weight_decay_rate'),
    Real(low=0.1, high=0.5, name='droprate'),
    # Integer(low=2, high=10, name='T_max'),
    Categorical(["epsilon", "KNN"], name="phy_top_type"),
    Categorical(["epsilonNN", "KNN"], name="graph_type"),
    Real(0, 1, name='graph_skip_conn'),
    Real(0, 1, name='epsilon'),
    Integer(low=1, high=10, name='top_k'),
    Integer(low=1, high=10, name='num_per'),
]

cnt = 0


def my_callback(res):
    global cnt
    cnt += 1
    logger.info(f"**************")
    logger.info(f"Iter {cnt}:")
    logger.info(f"Best Loss: {res['fun']}")
    logger.info(
        f"Best lr: {res['x'][0]},Best weight_decay_rate: {res['x'][1]},Best droprate: {res['x'][2]}, Best phy_top_type: {res['x'][3]},Best graph_type: {res['x'][4]},Best graph_skip_conn: {res['x'][5]}, Best epsilon: {res['x'][6]},Best top_k: {res['x'][7]}, Best num_per: {res['x'][8]}")
    logger.info(f"Now Loss: {res['func_vals'][-1]}")
    logger.info(
        f"Now lr: {res['x_iters'][-1][0]},Now weight_decay_rate: {res['x_iters'][-1][1]},Now droprate: {res['x_iters'][-1][2]},Now phy_top_type: {res['x_iters'][-1][3]},Now graph_type: {res['x_iters'][-1][4]},Now graph_skip_conn: {res['x_iters'][-1][5]}, Now epsilon: {res['x_iters'][-1][6]},Now top_k: {res['x_iters'][-1][7]}, Now num_per: {res['x_iters'][-1][8]}")


def objective(params):
    lr, weight_decay_rate, droprate, phy_top_type, graph_type, graph_skip_conn, epsilon, top_k, num_per = params
    # args, device, blocks = get_parameters()
    args.lr = lr
    args.weight_decay_rate = weight_decay_rate
    args.droprate = droprate
    args.phy_top_type = phy_top_type
    args.graph_type = graph_type
    args.graph_skip_conn = graph_skip_conn
    args.epsilon = epsilon
    args.top_k = top_k
    args.num_per = num_per

    data = ['data_steam.pickle', 'data_air.pickle', 'data_ele.pickle', 'data_c1.pickle', 'data_c2.pickle']
    score = ['score_steam.pkl', 'score_air.pkl', 'score_ele.pkl', 'score_c1.pkl', 'score_c2.pkl']
    score_list, train_iter_list, val_iter_list, test_iter_list = load_data(args, data, score)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)

    all_loss = search_train(score_list, loss, args, optimizer, scheduler, es, model, train_iter_list, val_iter_list,
                            test_iter_list)
    return all_loss


if __name__ == "__main__":
    args, device, blocks = get_parameters()

    '''
    data_preparate_steam("./data/dataset/Steam/Flow", "./data/dataset/Steam/Pressure", "./data/dataset/Steam/Temperature")
    data_preparate_ele("./data/dataset/Electricity/Angle", "./data/dataset/Electricity/Magnitude")
    data_preparate_air("./data/dataset/Compressed_Air/Flow", "./data/dataset/Compressed_Air/Pressure")
    data_preparate_c1("./data/dataset/Generator/Active_Power", "./data/dataset/Generator/Reactive_Power")
    data_preparate_c2("./data/dataset/Compressor_Power/Compressor_Power")


    '''
    data = ['data_steam.pickle', 'data_air.pickle', 'data_ele.pickle', 'data_c1.pickle', 'data_c2.pickle']
    score = ['score_steam.pkl', 'score_air.pkl', 'score_ele.pkl', 'score_c1.pkl', 'score_c2.pkl']
    score_list, train_iter_list, val_iter_list, test_iter_list = load_data(args, data, score)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)
    # model.load_state_dict(torch.load(f'checkpoint/model_epoch_20241005_133651_14.pth'))
    train(score_list, loss, args, optimizer, scheduler, es, model, train_iter_list, val_iter_list)
    test(score_list, loss, model, test_iter_list, args, logger)

    # result = gp_minimize(objective, search_space, n_calls=20, random_state=args.seed, callback=my_callback)
