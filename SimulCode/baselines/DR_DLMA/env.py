import numpy as np
import random

TDMA_delay = 8  # 设置TDMA延迟
ALOHA_delay = 6  # 设置ALOHA延迟
DRL_delay = 5  # 设置DRL延迟
delay_tag_TDMA = TDMA_delay - DRL_delay  # 计算延迟标签
delay_tag_ALOHA = ALOHA_delay - DRL_delay
DRL_action_length = 30  # DRL动作长度
TDMA_counter = 0  # 初始化TDMA计数器


class ENVIRONMENT(object):
    """环境类，用于模拟TDMA和DRL节点的行为"""

    def __init__(self, state_size=10):
        self.state_size = state_size  # 状态大小
        self.action_space = ['w', 't']  # 动作空间：等待（w）/传输（t）
        self.n_actions = len(self.action_space)  # 动作数量
        self.n_nodes = 3  # 节点数量
        self.TDMA_delay = TDMA_delay  # TDMA延迟
        self.ALOHA_delay = ALOHA_delay
        self.DRL_delay = DRL_delay  # DRL延迟
        self.delay_tag_tdma = delay_tag_TDMA  # 延迟标签
        self.delay_tag_aloha = delay_tag_ALOHA
        self.agent_reward_list = []  # 记录代理奖励的列表
        self.tdma_reward_list = []  # 记录TDMA奖励的列表
        self.aloha_reward_list = []
        self.reward_list = []  # 记录总奖励的列表
        self.DRL_observation_list = []  # 记录DRL节点观察的列表
        self.observation_reward_counter = 0  # 观察奖励计数器

        self.tdma_action_list = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  # 动作列表，表示TDMA节点的传输行为
        # ALOHA的发送概率
        self.action_pos = 0.3
        #self.aloha_action_list = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
        self.aloha_action_list = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]

    # 重置状态
    def reset(self):
        init_state = np.zeros(self.state_size, int)  # 初始化状态为全零数组
        return init_state

    # 执行动作并返回新状态、奖励等
    def step(self, action):
        global TDMA_counter  # 使用全局TDMA计数器
        ALOHA_action = 0
        tdma_reward = 0
        agent_reward = 0
        aloha_reward = 0
        reward = 0
        observation_ = 0

        # TDMA
        if self.delay_tag_tdma > 0:  # 当TDMA节点距离AP比DRL节点远时
            if self.tdma_action_list[TDMA_counter - self.delay_tag_tdma] == 1:
                TDMA_action = 1  # TDMA节点选择传输
            else:
                TDMA_action = 0  # TDMA节点选择等待
        else:
            if self.tdma_action_list[TDMA_counter - self.delay_tag_tdma - 10] == 1:
                TDMA_action = 1  # TDMA节点选择传输
            else:
                TDMA_action = 0  # TDMA节点选择等待

        # 生成ALOHA动作
        if self.delay_tag_aloha > 0:
            if self.aloha_action_list[TDMA_counter - self.delay_tag_aloha] == 1:
                if self.action_pos < random.random():
                    ALOHA_action = 1
                else:
                    ALOHA_action = 0
        else:
            if self.aloha_action_list[TDMA_counter - self.delay_tag_aloha - 10] == 1:
                if self.action_pos < random.random():
                    ALOHA_action = 1
                else:
                    ALOHA_action = 0

        # 根据tdma节点、Aloha节点和drl节点的动作，判断是否冲突，得到观测、奖励
        if action == 1:  # DRL节点选择传输
            if TDMA_action == 1 or ALOHA_action == 1:
                observation_ = -1  # 碰撞，传输不成功
            else:
                reward = 1
                agent_reward = 1
                observation_ = 1  # 传输成功

        else:  # DRL节点选择等待
            if TDMA_action == 1 and ALOHA_action == 0:
                reward = 1
                tdma_reward = 1
                observation_ = 1  # TDMA传输成功
            elif ALOHA_action == 1 and TDMA_action == 0:
                reward = 1
                aloha_reward = 1
                observation_ = 1  # ALOHA传输成功
            else:
                observation_ = 0

        TDMA_counter += 1  # 更新TDMA计数器
        if TDMA_counter == len(self.tdma_action_list):
            TDMA_counter = 0  # 重置TDMA计数器

        self.agent_reward_list.append(agent_reward)  # 记录智能体奖励
        self.tdma_reward_list.append(tdma_reward)  # 记录TDMA奖励
        self.aloha_reward_list.append(aloha_reward)
        self.reward_list.append(reward)  # 记录总奖励
        self.DRL_observation_list.append(observation_)  # 记录DRL节点的观察
        # 当时隙索引大于2*DRL的传播延迟后，才开始返回真实的奖励等信息，否则返回[0,0,0,0]
        self.observation_reward_counter += 1  # 更新观察奖励计数器
        if self.observation_reward_counter >= 2 * self.DRL_delay:
            counter11 = self.observation_reward_counter - (2 * self.DRL_delay)
            return self.DRL_observation_list[counter11], self.reward_list[counter11], self.agent_reward_list[counter11], \
                self.tdma_reward_list[counter11], self.aloha_reward_list[counter11]
        else:
            return 0, 0, 0, 0, 0  # 返回默认值
