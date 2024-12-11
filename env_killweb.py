import torch
import numpy as np

class Env_Kill_Web():

    def __init__(self, cfg):
        self.batch = cfg.batch
        self.n_blue_device = cfg.n_blue_device # 我方设备
        self.n_red_device = cfg.n_red_device # 敌方设备

    def get_blue_nodes(self, seed=None):
        """
        return nodes: (n_blue_device, 17)
        side 1 [0 | 1] 对于我方设备，恒为0
        type 3 [0 | 1]
        capability 4 float
        location 3 float
        time 1 float
        radius 1 float 对于我方设备，恒为0
        n_channel 4 int
        """
        if seed is not None:
            torch.manual_seed(seed)
        side = torch.zeros(size=(self.n_blue_device, 1), dtype=torch.int)
        type = torch.randint(0, 2, size=(self.n_blue_device, 3), dtype=torch.int)
        clt = torch.rand(size=(self.n_blue_device, 8), dtype=torch.float32)
        radius = torch.zeros(size=(self.n_blue_device, 1), dtype=torch.float32)
        n_channel = torch.randint(1, 10, size=(self.n_blue_device, 4), dtype=torch.int)
        return torch.cat([side, type, clt, radius, n_channel], dim=-1)

    def get_red_nodes(self, seed=None):
        """
        return nodes: (n_blue_device, 17)
        side 1 [0 | 1] int 对于敌方设备，恒为1
        type 3 [0 | 1] int 对于敌方设备，恒为0
        capability 4 float 对于敌方设备，恒为0
        location 3 float
        time 1 float
        radius 1 float
        n_channel 4 int 对于敌方设备，恒为0
        """
        if seed is not None:
            torch.manual_seed(seed)
        side = torch.ones(size=(self.n_red_device, 1), dtype=torch.int)
        type = torch.zeros(size=(self.n_red_device, 3), dtype=torch.int)
        capability = torch.zeros(size=(self.n_red_device, 4), dtype=torch.float32)
        ltr = torch.rand(size=(self.n_red_device, 5), dtype=torch.float32)
        n_channel = torch.zeros(size=(self.n_red_device, 4), dtype=torch.int)
        return torch.cat([side, type, capability, ltr, n_channel], dim=-1)

    def stack_blue_nodes(self, seed=None):
        """
        return nodes: (n_samples, n_blue_device, 17)
        """
        list = [self.get_blue_nodes() for _ in range(self.batch)]
        inputs = torch.stack(list, dim=0)
        return inputs

    def stack_red_nodes(self, seed=None):
        """
        return nodes: (n_samples, n_blue_device, 17)
        """
        list = [self.get_red_nodes() for _ in range(self.batch)]
        inputs = torch.stack(list, dim=0)
        return inputs

    def get_batch_blue_device(self, n_samples, seed=None):
        """
        return nodes: (n_samples, n_blue_device, 17)
        """
        if seed is not None:
            torch.manual_seed(seed)
        side = torch.zeros(size=(n_samples, self.n_blue_device, 1), dtype=torch.int)
        type = torch.randint(0, 2, size=(n_samples, self.n_blue_device, 3), dtype=torch.int)
        clt = torch.rand(size=(n_samples, self.n_blue_device, 8), dtype=torch.float32)
        radius = torch.zeros(size=(n_samples, self.n_blue_device, 1), dtype=torch.float32)
        n_channel = torch.randint(1, 10, size=(n_samples, self.n_blue_device, 4), dtype=torch.int)
        return torch.cat([side, type, clt, radius, n_channel], dim=-1)

    # def get_batch_blue_device(self, n_samples, seed=None):
    #     """
    #     return nodes: (n_samples, n_blue_device, 3)
    #     """
    #     if seed is not None:
    #         torch.manual_seed(seed)
    #     return torch.rand(size=(n_samples, self.n_blue_device, 3), dtype=torch.float32)

    def get_batch_red_device(self, n_samples, seed=None):
        """
        return nodes: (n_samples, n_red_device, 17)
        """
        if seed is not None:
            torch.manual_seed(seed)
        side = torch.ones(size=(n_samples, self.n_red_device, 1), dtype=torch.int)
        type = torch.zeros(size=(n_samples, self.n_red_device, 3), dtype=torch.int)
        capability = torch.zeros(size=(n_samples, self.n_red_device, 4), dtype=torch.float32)
        ltr = torch.rand(size=(n_samples, self.n_red_device, 5), dtype=torch.float32)
        n_channel = torch.zeros(size=(n_samples, self.n_red_device, 4), dtype=torch.int)
        return torch.cat([side, type, capability, ltr, n_channel], dim=-1)


    def get_random_solution(self, nodes):
        """
        生成解
        :param nodes: (n_blue_device, 17)
        return solution: (3, )
        """
        solution = []
        # 选择侦察设备
        city = np.random.randint(self.n_blue_device)
        while nodes[city, 1].item() != 1:
            city = np.random.randint(self.n_blue_device)
        solution.append(city)
        # 选择指控设备
        city = np.random.randint(self.n_blue_device)
        while nodes[city, 2].item() != 1:
            city = np.random.randint(self.n_blue_device)
        solution.append(city)
        # 选择打击设备
        city = np.random.randint(self.n_blue_device)
        while nodes[city, 3].item() != 1:
            city = np.random.randint(self.n_blue_device)
        solution.append(city)
        return torch.from_numpy(np.array(solution))


    def cal_distance(self, inputs, red_device, tours):
        """
        *** this function is faster version of stack_l! ***
        inputs: (batch, n_blue_devices, 17), 蓝方设备
        red_device: (batch, 1, 17), 红方设备
        tours: (batch, 3), 预测路径
        d1: (batch, 3, 3)
        d2: (batch, 1, 3)
        d: (batch, 4, 3)
        """
        d1 = torch.gather(input=inputs[:, :, 8:11], dim=1, index=tours[:, :, None].repeat(1, 1, 3))
        # d2 = red_device[:, :, 8:11]
        # d = torch.cat([d1, d2], dim=1)
        d = d1
        return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
                + (d[:, 0] - d[:, -1]).norm(p=2, dim=1))  # distance from last node to first selected node)

        # d = torch.gather(input=inputs, dim=1, index=tours[:, :, None].repeat(1, 1, 3))
        # return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
        #         + (d[:, 0] - d[:, -1]).norm(p=2, dim=1))  # distance from last node to first selected node)







