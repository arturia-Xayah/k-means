import math
import random
from copy import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

deviation = 0.02  # 画线偏移量


class watermelon:
    num = 0  # 属性数量
    numbers = 0  # 数据个数
    attributes = []  # 属性值
    attributes_list = []  # 全部属性值列表
    family_list = []  # 初始向量集族
    vector_data = []  # 以向量格式存储的数据
    start_list = []  # 均值向量集

    best_family_list = []  # 最优集族
    best_start_list = []  # 最优均值向量
    best_grq = []  # 各个k值下的轮廓系数
    k_num = []  # k值列表
    max_num = -1  # 最大轮廓系数时的k
    distance = []  # 各个点之间的距离矩阵

    def __init__(self, num):
        self.deep = 0
        self.num = num
        self.numbers = 0
        for i in range(num):
            self.attributes.append([])
            self.attributes_list.append([])
            # self.attributes_p.append([])

    def add_data(self, *args):
        # print(args)
        for i in range(self.num):
            self.attributes[i].append(args[i])
            if not isinstance(args[i], str):
                continue
            elif args[i] not in self.attributes_list[i]:
                self.attributes_list[i].append(args[i])
                # self.attributes_p.append([])

        self.numbers += 1

    def show_data(self):
        print("属性数量:" + str(self.num))
        print("数据个数:" + str(self.numbers))
        for i in range(self.numbers):
            for j in range(self.num):
                print(self.attributes[j][i], end=" ")
            print('')
        print("属性值:")

        for i in range(self.num):
            print("第%d个属性的取值：" % (i + 1), end=' ')
            print(self.attributes_list[i])

    def trans_data(self):  # 将数据格式转换为向量
        for i in range(self.numbers):
            self.vector_data.append((self.attributes[0][i], self.attributes[1][i]))

    def cal_distance(self, start, end):  # 计算两个元素之间的欧氏距离
        return math.sqrt(pow(start[0] - end[0], 2) + pow(start[1] - end[1], 2))

    def vector_avg(self, *args):  # 求向量均值
        length = float(len(args))
        data_0 = 0
        data_1 = 0
        for data in args:
            data_0 += data[0]
            data_1 += data[1]
        return data_0 / length, data_1 / length

    def ana_distance(self):  # 计算距离矩阵
        self.vector_data.clear()
        self.distance.clear()
        for i in self.family_list:
            self.vector_data += i
        # print(self.vector_data)
        for i in range(self.numbers):
            for j in range(i):
                self.distance.append(self.cal_distance(self.vector_data[i], self.vector_data[j]))

    def find_distance(self, start, end):  # 下标从0开始,小号在前
        if start == end:
            return 0
        if start > end:  # 变量交换
            start, end = end, start
        return self.distance[int(end * (end - 1) / 2 + start)]

    def k_means(self, train_epoch, k_list):  # start_num  初始样本数量
        # self.k_num=[0 for x in range(len(k_list))]
        # self.best_grq=[0 for x in range(len(k_list))]
        all_best_family_list = []
        all_best_start_list = []
        for start_num in k_list:
            temp_best_grq = 0.0
            temp_best_family_list = []
            temp_best_start_list = []
            for now_epoch in range(train_epoch):
                self.family_list.clear()
                self.start_list.clear()
                if start_num > self.numbers:
                    print("初始样本数不得大于数据个数！")
                    print("当前数据个数:%d,初始样本数:%d" % (self.numbers, start_num))

                self.start_list = [self.vector_data[x] for x in
                                   random.sample(range(0, self.numbers), start_num)]  # start_list为均值向量集合

                epoch = 0  # 轮数
                while 1:
                    epoch += 1
                    self.family_list = [[] for x in range(start_num)]  # 初始化样本集簇
                    # print(start_list)
                    for i in range(len(self.vector_data)):  # 将各个样本加入最近的均值向量的簇
                        now_data = self.vector_data[i]  # 当前样本
                        min_val = 99999
                        min_data = 0
                        for j in self.start_list:
                            now_val = self.cal_distance(now_data, j)
                            if now_val < min_val:
                                min_val = now_val
                                min_data = j
                        self.family_list[self.start_list.index(min_data)].append(now_data)

                    is_update = 0  # 判断是否进行了均值向量的更新
                    for i in range(start_num):
                        new_val = self.vector_avg(*tuple(self.family_list[i]))  # 新的均值向量
                        if new_val != self.start_list[i]:  # 均值向量不相等
                            self.start_list[i] = new_val
                            is_update = 1

                    # self.draw_pic(epoch)
                    if not is_update:
                        break

                self.ana_distance()  # 计算距离矩阵
                silhouette_coefficient = 0.0  # 轮廓系数和
                point_list = [[] for x in range(start_num)]  # 存储各个类的点序号
                count = 0
                for i in range(start_num):  # 数据预处理
                    for j in range(len(self.family_list[i])):
                        point_list[i].append(count)
                        count += 1
                for i in range(start_num):
                    for j in point_list[i]:
                        temp_in = 0.0  # 类内距离和
                        for k in point_list[i]:
                            temp_in += self.find_distance(j, k)
                        if len(point_list[i]) > 1:
                            temp_in /= len(point_list[i]) - 1
                        temp_out = 0.0  # 类外距离和
                        min_temp_out = 99999.0  # 最小类外距离和
                        for k in range(start_num):  # 查找不同类
                            temp_out = 0
                            if k == i:
                                break
                            else:
                                for kk in point_list[k]:
                                    temp_out += self.find_distance(j, kk)
                                temp_out /= len(point_list[k])
                            if temp_out < min_temp_out:
                                min_temp_out = temp_out

                        silhouette_coefficient += (min_temp_out - temp_in) / max(min_temp_out, temp_in)
                silhouette_coefficient /= self.numbers  # 求得轮廓系数
                # print("轮廓系数:%.4f" % silhouette_coefficient)
                if silhouette_coefficient > temp_best_grq:  # 当前K值中的最佳效果
                    temp_best_start_list = deepcopy(self.start_list)
                    temp_best_family_list = deepcopy(self.family_list)
                    temp_best_grq = silhouette_coefficient

            self.k_num.append(start_num)
            self.best_grq.append(temp_best_grq)
            print("当前k值：%d，最佳轮廓系数：%.4f" % (start_num, temp_best_grq))
            if self.max_num == -1 or temp_best_grq > self.best_grq[self.max_num]:
                self.max_num = len(self.k_num) - 1
                self.best_start_list = deepcopy(temp_best_start_list)
                self.best_family_list = deepcopy(temp_best_family_list)

    def draw_pic(self, num):

        self.start_list = self.best_start_list  # 数据重新装填
        self.family_list = self.best_family_list
        x1min, x1max = min(self.attributes[0]), max(self.attributes[0])
        x2min, x2max = min(self.attributes[1]), max(self.attributes[1])
        # x1min, x2min, x1max, x2max = 0.1, 0, 0.9, 0.8
        x1_l, x1_h = x1min - (x1max - x1min) * 0.2, x1max + (x1max - x1min) * 0.2
        x2_l, x2_h = x2min - (x2max - x2min) * 0.2, x2max + (x2max - x2min) * 0.2

        x1_l, x1_h, x2_l, x2_h = 0.1, 0.9, 0, 0.8
        plt.xlim(x1_l, x1_h)
        plt.ylim(x2_l, x2_h)
        x1, x2 = np.linspace(x1_l, x1_h, 100), np.linspace(x2_l, x2_h, 100)
        X1, X2 = np.meshgrid(x1, x2)  # 矩阵网格
        plt.title("第" + str(num) + "轮迭代后")
        plt.xlabel("密度")
        plt.ylabel("含糖量")

        color_list = ['b', 'y', 'g', 'b', 'y', 'g', 'b', 'y', 'g', 'b', 'y', 'g']
        for i in range(len(self.start_list)):
            points = [[], []]
            edge_point = []  # 边缘点

            for j in self.family_list[i]:
                points[0].append(j[0])
                points[1].append(j[1])
                edge_point.append([j[0], j[1], 0])
            plt.scatter(points[0], points[1], c=color_list[i], s=80, label='类别' + str(i + 1))

            # 计算凸包
            putout = melkman_algorithm(edge_point)
            points = [[], []]
            for j in putout:
                length = math.sqrt(pow(j[1] - self.start_list[i][1], 2) + pow(j[0] - self.start_list[i][0], 2))
                cos = (j[0] - self.start_list[i][0]) / length
                sin = (j[1] - self.start_list[i][1]) / length
                points[0].append(j[0] + cos * deviation)
                points[1].append(j[1] + sin * deviation)
            length = math.sqrt(
                pow(putout[0][1] - self.start_list[i][1], 2) + pow(putout[0][0] - self.start_list[i][0], 2))
            cos = (putout[0][0] - self.start_list[i][0]) / length
            sin = (putout[0][1] - self.start_list[i][1]) / length
            points[0].append(putout[0][0] + cos * deviation)
            points[1].append(putout[0][1] + sin * deviation)
            plt.plot(points[0], points[1], color='r', linewidth=3, linestyle='--')
        points = [[], []]
        for i in self.start_list:
            points[0].append(i[0])
            points[1].append(i[1])
        plt.scatter(points[0], points[1], c='r', marker='+', s=80, label='均值向量')
        plt.legend()
        plt.show()

        # 画轮廓系数变化图
        plt.title('轮廓系数随K值的变换情况')
        # plt.plot(range(1, len(lost) + 1), lost, 'o--', markersize=2, label='lost')
        plt.plot(self.k_num, self.best_grq, 'o--', markersize=2, label='轮廓系数')
        # plt.plot([1, len(lost)], [1 / m, 1 / m], 'k-.', linewidth=0.3, label='1/m')
        plt.xlabel('K值')
        plt.ylabel('轮廓系数')
        plt.legend()
        plt.show()


def take_twice(my_list):
    return my_list[2]


def is_in_left(my_list, point):
    now = len(my_list) - 2
    last = len(my_list) - 1
    vector_1 = (my_list[last][0] - my_list[now][0], my_list[last][1] - my_list[now][1])
    vector_2 = (point[0] - my_list[now][0], point[1] - my_list[now][1])
    if vector_1[0] * vector_2[1] - vector_2[0] * vector_1[1] < 0:  # x1*y2-x2*y1
        return 0
    return 1


def melkman_algorithm(data_list):  # melkman凸包算法
    putout_list = []  # 最终凸包列表
    waiting_list = []
    waiting_list_positive = []  # 正斜率
    waiting_list_negative = []  # 负斜率
    aim_data = data_list[0]  # 存储y轴最小的向量
    for one_data in data_list[1:]:
        if one_data[1] < aim_data[1]:  # 当前向量y坐标小于存储的数据
            waiting_list.append(aim_data)
            aim_data = one_data
            continue
        elif one_data[1] == aim_data[1]:
            if one_data[0] < aim_data[0]:  # y轴相同，若x轴更小
                waiting_list.append(aim_data)
                aim_data = one_data
                continue
        waiting_list.append(one_data)

    putout_list.append(aim_data)  # 初始点放入栈
    for i in range(len(waiting_list)):  # 计算其他点与目标点的斜率
        waiting_list[i][2] = (waiting_list[i][1] - aim_data[1]) / (waiting_list[i][0] - aim_data[0])
        if waiting_list[i][2] > 0:
            waiting_list_positive.append(waiting_list[i])
        else:
            waiting_list_negative.append(waiting_list[i])
    waiting_list_positive.sort(key=take_twice, reverse=True)
    waiting_list_negative.sort(key=take_twice, reverse=True)
    waiting_list = waiting_list_negative + waiting_list_positive
    if len(waiting_list) <= 2:
        while waiting_list:
            putout_list.append(waiting_list.pop())
        return putout_list
    else:
        putout_list.append(waiting_list.pop())
        putout_list.append(waiting_list.pop())

    while waiting_list:
        point = waiting_list.pop()
        if not is_in_left(putout_list, point):  # 该点不在凸包内
            while not is_in_left(putout_list, point):
                putout_list.pop()
        putout_list.append(point)
    return putout_list


if __name__ == '__main__':
    data = pd.read_csv('watermelon4_0_Ch.txt', index_col=None, header=None)
    print(data)
    print(data.shape[1])
    print(data[0][1])
    w = watermelon(data.shape[1])
    for i in range(data.shape[0]):  # 数据读取
        temp_data = []
        for j in range(w.num):
            temp_data.append(data[j][i])
        w.add_data(*tuple(temp_data))
    # w.show_data()
    w.trans_data()
    w.k_means(100, [2, 3, 4, 5])
    print("最优k值：%d,轮廓系数:%f" % (w.k_num[w.max_num], w.best_grq[w.max_num]))
    w.draw_pic("最终")
    # w.draw_pic("最终")
