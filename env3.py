# import pandas as pd
# import numpy as np
# import sqlite3  # 根据您的数据库类型选择合适的库
#
# # 连接到数据库
# conn = sqlite3.connect('eicu_database.db')
#
# # 从数据库中读取数据
# patients_df = pd.read_sql_query("SELECT * FROM patient", conn)
# infusion_drug_df = pd.read_sql_query("SELECT * FROM infusiondrug", conn)
# vital_period_df = pd.read_sql_query("SELECT * FROM vitalperdioc", conn)
#
# # 关闭数据库连接
# conn.close()
#
# # 数据预处理
# # 将时间转换为实际的时间戳
# patients_df['hospitaladmittime'] = patients_df['hospitaladmitoffset'] - patients_df['hospitaladmitoffset']
# patients_df['hospitaldischargetime'] = patients_df['hospitaldischargeoffset'] - patients_df['hospitaladmitoffset']
#
# # 为每个病人创建状态和动作
# # 这里我们需要实现从 infusion_drug_df 和 vital_period_df 中提取最后一次观察数据的逻辑
# # 并结合患者信息生成状态
#
#
# import gym
# from gym import spaces
#
#
# class PatientEnv(gym.Env):
#     def __init__(self, patients, infusion_drugs, vitals):
#         super(PatientEnv, self).__init__()
#
#         # 保存数据
#         self.patients = patients
#         self.infusion_drugs = infusion_drugs
#         self.vitals = vitals
#
#         # 动作空间：假设我们有 3 种药物
#         self.action_space = spaces.Discrete(3)
#
#         # 状态空间：我们根据观察指标构建状态
#         self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
#
#         # 其他变量
#         self.current_step = 0
#         self.current_patient = None
#
#     def reset(self):
#         # 重置环境
#         self.current_step = 0
#         self.current_patient = self.patients.sample().iloc[0]  # 随机选择一个病人
#         return self._get_observation()
#
#     def step(self, action):
#         # 执行动作并返回新的状态和奖励
#         self.current_step += 1
#
#         # 根据当前病人的状态和选择的动作计算奖励
#         # 这里需要定义如何计算奖励
#
#         observation = self._get_observation()
#         reward = self._calculate_reward(action)
#         done = self.current_step >= 100  # 设定结束条件
#
#         return observation, reward, done, {}
#
#     def _get_observation(self):
#         # 从 vitals 和 infusion_drugs 中提取观察数据
#         # 返回状态作为观察
#         return np.array([self.current_patient['heartrate'], self.current_patient['respiration'], ...])
#
#     def _calculate_reward(self, action):
#         # 计算奖励
#         return reward
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# class DQN(nn.Module):
# def __init__(self, state_size, action_size):
# super(DQN, self).__init__()
# self.fc1 = nn.Linear(state_size, 24)
# self.fc2 = nn.Linear(24, 24)
# self.fc3 = nn.Linear(24, action_size)
#
# def forward(self, x):
# x = torch.relu(self.fc1(x))
# x = torch.relu(self.fc2(x))
# return self.fc3(x)
#
# import random
# from collections import deque
#
#
# def train_dqn(env):
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     dqn = DQN(state_size, action_size)
#     optimizer = optim.Adam(dqn.parameters(), lr=0.001)
#     loss_fn = nn.MSELoss()
#
#     memory = deque(maxlen=2000)
#     batch_size = 32
#
#     episodes = 1000
#     for e in range(episodes):
#         state = env.reset()
#         total_reward = 0
#
#         for time in range(500):
#             action = env.action_space.sample()  # 使用 epsilon-greedy 策略
#
#             next_state, reward, done, _ = env.step(action)
#             memory.append((state, action, reward, next_state, done))
#             state = next_state
#             total_reward += reward
#
#             if done:
#                 break
#
#             # 从记忆中采样并训练
#             if len(memory) > batch_size:
#                 minibatch = random.sample(memory, batch_size)
#                 for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
#                     target = m_reward + (1 - m_done) * 0.99 * torch.max(dqn(torch.FloatTensor(m_next_state)))
#                     target_f = dqn(torch.FloatTensor(m_state))
#                     target_f[m_action] = target
#                     optimizer.zero_grad()
#                     loss = loss_fn(target_f, dqn(torch.FloatTensor(m_state)))
#                     loss.backward()
#                     optimizer.step()
#
#         print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")
#
#
# # 创建环境并训练模型
# env = PatientEnv(patients_df, infusion_drug_df, vital_period_df)
# train_dqn(env)
#
#
# def evaluate_model(env, model):
#     state = env.reset()
#     done = False
#     total_reward = 0
#
#     while not done:
#         action = model(torch.FloatTensor(state)).argmax().item()  # 选择最佳动作
#         next_state, reward, done, _ = env.step(action)
#         state = next_state
#         total_reward += reward
#
#     print(f"Total Reward: {total_reward}")
#
#
# # 使用训练好的模型进行评估
# evaluate_model(env, dqn)
#
#
