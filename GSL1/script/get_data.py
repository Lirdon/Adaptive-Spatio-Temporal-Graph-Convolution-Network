import pandas as pd
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#from torch_geometric.data import DataLoader
import numpy as np
import os


'''
def get_all_excel_data(folder_path):
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]  # 获取文件夹中的所有Excel文件
    data_list = []
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)  # 读取每个Excel文件的数据
        data_list.append(df)

        #df_processed = df.apply(lambda col: process_column_with_stable(col))
        #data_list.append(df_processed)
    stacked_data = np.stack(data_list, axis=0)  # 堆叠数据
    return stacked_data
'''


def fill_columns_to_max_length(df, max_length):
    # 获取当前DataFrame的行数
    current_length = len(df)
    # 如果当前长度已经等于或超过max_length，则无需填充
    if current_length >= max_length:
        return df

    # 计算还需要多少行才能达到max_length
    additional_rows_needed = max_length - current_length

    # 使用最后一行的值创建一个新的DataFrame，行数为需要添加的行数
    last_values_df = pd.DataFrame([df.iloc[-1].tolist()] * additional_rows_needed,
                                  columns=df.columns)

    # 将原来的DataFrame与新创建的行拼接起来
    result_df = pd.concat([df, last_values_df], ignore_index=True)
    return result_df

def get_all_excel_data(folder_path):
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]  # 获取文件夹中的所有Excel文件
    data_list = []

    # 第一遍读取文件以确定最大行数
    max_rows = 0
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)  # 读取每个Excel文件的数据
        max_rows = max(max_rows, df.shape[0])  # 更新最大行数
        data_list.append(df)

    # 第二遍填充列到最大长度并转换为NumPy数组
    stacked_data = []
    for df in data_list:
        df_filled = fill_columns_to_max_length(df, max_rows)  # 填充列到最大长度

        stacked_data.append(df_filled.to_numpy())  # 转换为NumPy数组

    # 堆叠数据，形成三维数组
    stacked_data = np.stack(stacked_data, axis=0)  # 堆叠数据
    return stacked_data


def get_node_data_steam(path_G, path_P, path_T):
    G_df = get_all_excel_data(path_G)   #通过转换确保G——df是kg/s
    P_df = get_all_excel_data(path_P) * 1e6   #Pa
    #P_df = [df * 1e6 for df in P_df]
    T_df = get_all_excel_data(path_T) + 273.15    #K
    #T_df = [df +273.15 for df in T_df]
    return G_df, P_df, T_df

def get_node_data_c1(path_G, path_P):
    G_df = get_all_excel_data(path_G)
    P_df = get_all_excel_data(path_P)
    return G_df, P_df

def get_node_data_c2(path_G):
    G_df = get_all_excel_data(path_G)
    return G_df

def get_node_data_air(path_G, path_P):
    G_df = get_all_excel_data(path_G)
    P_df = get_all_excel_data(path_P) * 1e6
    #P_df = [df * 1e6 for df in P_df]
    return G_df, P_df

def get_node_data_ele(path_G, path_P):
    G_df = get_all_excel_data(path_G)
    P_df = get_all_excel_data(path_P)
    return G_df, P_df


def load_data(dataset_name, len_train, len_val):
    train = dataset_name[: len_train]
    val = dataset_name[len_train: len_train + len_val]
    test = dataset_name[len_train + len_val:]
    return train, val, test


def data_split(S_data_x, point): #edge_index, S1, S2, t):
    data_list = []

    for i in range(S_data_x.shape[0]):
        x = S_data_x[i]  # [2, 12, 24] - 节点特征
        y = point[i]

        # 创建Data对象
        #data = Data(x=x, edge_index=edge_index, edge_attr=S1, y=y)
        data = Data(x=x, y=y)
        data_list.append(data)
    return data_list
