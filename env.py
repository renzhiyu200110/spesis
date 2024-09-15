import gym
from gym import spaces
import numpy as np
import pandas as pd


class SepsisTreatmentEnv(gym.Env):
    def __init__(self, patient_data, infusion_data, vital_data):
        super(SepsisTreatmentEnv, self).__init__()

        # 数据加载
        self.patients = pd.read_csv(patient_data)  # 患者信息
        self.infusion_drugs = pd.read_csv(infusion_data)  # 注射药物信息
        self.vital_periods = pd.read_csv(vital_data)  # 生命体征信息

        # 设定状态空间
        # 状态包括心率（heart_rate）、血氧饱和度（sao2）和呼吸频率（respiration）
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(3,), dtype=np.float32)  # -1 用于表示缺失值

        # 动作空间
        self.action_space = spaces.Discrete(len(self.infusion_drugs))  # 药物数量

        # 记录当前状态和步数
        self.current_patient_id = None
        self.current_step = 0
        self.current_infusions = None

    def reset(self):
        # 随机选择一个病人
        self.current_patient_id = np.random.choice(self.patients['patient_id'])
        self.current_step = 0
        self.current_infusions = self.infusion_drugs[self.infusion_drugs['patient_id'] == self.current_patient_id]

        # 返回初始状态
        return self.get_state()

    def get_state(self):
        # 获取最近的观察数据
        if self.current_step == 0:
            # 如果是第一次重置，返回缺省状态
            return np.array([-1, -1, -1])  # 表示缺失的初始状态

        # 获取当前时间的注射药物信息
        infusion_time = self.current_infusions.iloc[self.current_step]['offset']

        # 找到距离最近的观察
        recent_vital = self.vital_periods[(self.vital_periods['patient_id'] == self.current_patient_id) &
                                          (self.vital['timestamp'] <= infusion_time)].iloc[-1]

        # 提取心率、血氧和呼吸率，并处理缺失值
        heart_rate = recent_vital['heart_rate'] if not pd.isnull(recent_vital['heart_rate']) else -1
        sao2 = recent_vital['sao2'] if not pd.isnull(recent_vital['sao2']) else -1
        respiration = recent_vital['respiration'] if not pd.isnull(recent_vital['respiration']) else -1

        return np.array([heart_rate, sao2, respiration])

    def step(self, action):
        # 根据选择的药物执行动作
        drug_name = self.infusion_drugs.iloc[action]['drug_name']
        drug_dose = self.infusion_drugs.iloc[action]['dose']

        # 更新当前步数
        self.current_step += 1

        # 获取当前状态
        if self.current_step < len(self.current_infusions):
            next_state = self.get_state()
            reward = self.calculate_reward(next_state)  # 计算奖励
            done = False
        else:
            next_state = np.array([-1, -1, -1])  # 终止状态
            reward = 0
            done = True

        return next_state, reward, done, {}

    def calculate_reward(self, state):
        # 奖励机制
        # 示例：心率正常（<100）和血氧饱和度正常（>90），给予正奖励
        heart_rate, sao2, respiration = state

        reward = 0
        if heart_rate != -1 and heart_rate < 100:
            reward += 1
        if sao2 != -1 and sao2 > 90:
            reward += 1
        if respiration != -1 and respiration < 20:  # 假设正常呼吸频率为20以下
            reward += 1

        return reward

    def render(self, mode='human'):
        # 可视化环境状态（可选）
        pass