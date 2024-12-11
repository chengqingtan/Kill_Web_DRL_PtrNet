import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm


def caculateTanceQuality(d, intelRadius):
    pm = 0.5  # 在距离探测范围时的探测概率
    pf = 0.01  # 在距离探测范围时的探测错误率
    if intelRadius == 0:
        return 0
    else:
        term1 = (d / intelRadius) ** 4
        numerator = term1 * np.log(pm) * np.log(pf)
        denominator = np.log(pf) - np.log(pm) + term1 * np.log(pm)
        return np.exp(numerator / denominator)

def caculateDajiQuality(distance, dajiCap, CEP, r):

    if distance <= dajiCap:
        return norm.cdf(r, loc=CEP, scale=CEP / 10)
    else:
        return 0

def caculateTongxinQuality(distance, tongXinMatrix):
    return tongXinMatrix**2 / (tongXinMatrix**2 + distance**2)

# 读取 Excel 文件
redData_TanceDaji = pd.read_excel('red_TanceDaji.xlsx', sheet_name='Sheet1', header=None, usecols='C:O', skiprows=2, nrows=60)
blueData_TanceDaji = pd.read_excel('blue_TanceDaji.xlsx', sheet_name='Sheet1', header=None, usecols='C:O', skiprows=2, nrows=37)
redData_TongXin = pd.read_excel('red_TanceDaji.xlsx', sheet_name='Sheet2', header=None, usecols='B:H', skiprows=1, nrows=10)
blueData_TongXin = pd.read_excel('blue_TanceDaji.xlsx', sheet_name='Sheet2', header=None, usecols='B:H', skiprows=1, nrows=10)

# 提取红色设备的相关数据
r_Number = redData_TanceDaji.shape[0]  # 红色设备数量
redXYZ = redData_TanceDaji.iloc[:, 0:3].values  # 红色设备的 XYZ 坐标
redID = redData_TanceDaji.iloc[:, 3].values  # 红色设备的 ID
redType = redData_TanceDaji.iloc[:, 4].values  # 红色设备的 Type
redTanCeCap = redData_TanceDaji.iloc[:, 5:8].values * 1000  # 红色设备的 TanCeCap，Km 转换为米
redTanCeChannel = redData_TanceDaji.iloc[:, 11].values  # 红色设备的 TanCeChannel
redDaJiCap = redData_TanceDaji.iloc[:, 8:11].values * 1000  # 红色设备的 DaJiCap，Km 转换为米
redDaJiChannel = redData_TanceDaji.iloc[:, 12].values  # 红色设备的 DaJiChannel

# 提取红色设备通信数据
redTongxinMatrix = redData_TongXin.iloc[0:7, 0:7].values * 1000  # 红色设备通信矩阵，Km 转换为米
redTongxinChannel = redData_TongXin.iloc[8, 0:7].values  # 红色设备的 TongxinChannel
redControl = redData_TongXin.iloc[9, 0:7].values  # 红色设备的 Control

# 提取蓝色设备的相关数据
b_Number = blueData_TanceDaji.shape[0]  # 蓝色设备数量
blueXYZ = blueData_TanceDaji.iloc[:, 0:3].values  # 蓝色设备的 XYZ 坐标
blueID = blueData_TanceDaji.iloc[:, 3].values  # 蓝色设备的 ID
blueType = blueData_TanceDaji.iloc[:, 4].values  # 蓝色设备的 Type
blueTanCeCap = blueData_TanceDaji.iloc[:, 5:8].values * 1000  # 蓝色设备的 TanCeCap，Km 转换为米
blueTanCeChannel = blueData_TanceDaji.iloc[:, 11].values  # 蓝色设备的 TanCeChannel
blueDaJiCap = blueData_TanceDaji.iloc[:, 8:11].values * 1000  # 蓝色设备的 DaJiCap，Km 转换为米
blueDaJiChannel = blueData_TanceDaji.iloc[:, 12].values  # 蓝色设备的 DaJiChannel

# 提取蓝色设备通信数据
blueTongxinMatrix = blueData_TongXin.iloc[0:7, 0:7].values * 1000  # 蓝色设备通信矩阵，Km 转换为米
blueTongxinChannel = redData_TongXin.iloc[8, 0:7].values  # 蓝色设备的 TongxinChannel
blueControl = redData_TongXin.iloc[9, 0:7].values  # 蓝色设备的 Control


# 计算装备之间的距离
distanceRB = np.sqrt(
    np.sum((blueXYZ[:, np.newaxis, :] - redXYZ[np.newaxis, :, :]) ** 2, axis=2)
)
print(distanceRB) #待验证

# 初始化 q_Tance 矩阵
q_Tance = np.zeros((b_Number, r_Number))


# 计算探测质量
for i in range(b_Number):
    for j in range(r_Number):
        if blueType[i] in [1, 2, 3]:
            q_Tance[i, j] = caculateTanceQuality(distanceRB[i, j], redTanCeCap[j, 0])
        elif blueType[i] == 4:
            q_Tance[i, j] = caculateTanceQuality(distanceRB[i, j], redTanCeCap[j, 1])
        else:
            q_Tance[i, j] = caculateTanceQuality(distanceRB[i, j], redTanCeCap[j, 2])


# 计算红方装备之间的距离
r_Distance = np.zeros((r_Number, r_Number))
for i in range(r_Number):
    for j in range(r_Number):
        r_Distance[i, j] = np.sqrt(
            (redXYZ[i, 0] - redXYZ[j, 0]) ** 2 +
            (redXYZ[i, 1] - redXYZ[j, 1]) ** 2 +
            (redXYZ[i, 2] - redXYZ[j, 2]) ** 2
        )


# 计算红方装备之间的直接通信概率
q_communication = np.zeros((r_Number, r_Number))
# 使用循环计算通信概率
for i in range(r_Number):
    for j in range(r_Number):
        q_communication[i, j] = caculateTongxinQuality(
            r_Distance[i, j],
            redTongxinMatrix[redType[i] - 1, redType[j] - 1]  # 注意索引从 0 开始
        )
print(q_communication)

# 计算红方装备之间的最优通信概率
q_communication_2 = -np.log(q_communication) # 将通信概率转换为负对数权重
shortestPath = {}
totalCost = np.zeros((r_Number, r_Number))
# 使用 NetworkX 计算最短路径
graph = nx.DiGraph()
# 构建图，节点为装备索引，边的权重为 `q_communication_2`
for i in range(r_Number):
    for j in range(r_Number):
        if i != j:  # 排除自身连接
            graph.add_edge(i, j, weight=q_communication_2[i, j])
# 计算每对节点之间的最短路径
for i in range(r_Number):
    for j in range(r_Number):
        if i != j:
            path = nx.shortest_path(graph, source=i, target=j, weight='weight')
            cost = nx.shortest_path_length(graph, source=i, target=j, weight='weight')
            shortestPath[(i, j)] = path
            totalCost[i, j] = cost
# 根据总成本计算最优通信概率
q_communication_opt = np.exp(-totalCost)
q_communication_opt_route = shortestPath
print(q_communication_opt)
print(shortestPath)

#计算红方装备指控质量
q_command = np.zeros((r_Number, r_Number))
r_CommandCap = [1, 1, 1, 1, 0, 0, 0]  # 红方命令能力
for i in range(r_Number):
    for j in range(r_Number):
        q_command[i, j] = q_communication_opt[i, j] * r_CommandCap[redType[i]]
print(q_command)

# 红方到蓝方装备的打击质量
q_Daji = np.zeros((r_Number, b_Number))
b_Radius = [35, 30, 15, 66, 10, 8, 9]
CEP_air, CEP_sea, CEP_land = 5, 10, 8  # 对空、对海、对陆的 CEP
for i in range(r_Number):  # 红方
    for j in range(b_Number):  # 蓝方
        if blueType[j] in [1, 2, 3]:  # 对空
            q_Daji[i, j] = caculateDajiQuality(distanceRB[j, i], redDaJiCap[i, 0], CEP_air, b_Radius[redType[j]])
        elif blueType[j] == 4:  # 对海
            q_Daji[i, j] = caculateDajiQuality(distanceRB[j, i], redDaJiCap[i, 1], CEP_sea, b_Radius[redType[j]])
        else:  # 对陆
            q_Daji[i, j] = caculateDajiQuality(distanceRB[j, i], redDaJiCap[i, 2], CEP_land, b_Radius[redType[j]])


print(q_Daji)