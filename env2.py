import numpy as np
import pandas as pd
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# 自定义环境
class SepsisTreatmentEnv(gym.Env):
    def __init__(self, patient_data, infusion_data, vital_data):
        super(SepsisTreatmentEnv, self).__init__()

        # 数据加载
        self.patients = pd.read_csv(patient_data)  # 患者信息
        self.infusion_drugs = pd.read_csv(infusion_data)  # 注射药物信息
        self.vital_periods = pd.read_csv(vital_data)  # 生命体征信息

        # 设定状态空间
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
        if self.current_step == 0:
            return np.array([-1, -1, -1])  # 表示缺失的初始状态

        infusion_time = self.current_infusions.iloc[self.current_step]['offset']
        recent_vital = self.vital_periods[(self.vital_periods['patient_id'] == self.current_patient_id) &
                                           (self.vital_periods['timestamp'] <= infusion_time)].iloc[-1]

        heart_rate = recent_vital['heart_rate'] if not pd.isnull(recent_vital['heart_rate']) else -1
        sao2 = recent_vital['sao2'] if not pd.isnull(recent_vital['sao2']) else -1
        respiration = recent_vital['respiration'] if not pd.isnull(recent_vital['respiration']) else -1

        return np.array([heart_rate, sao2, respiration])

    def step(self, action):
        drug_name = self.infusion_drugs.iloc[action]['drug_name']
        drug_dose = self.infusion_drugs.iloc[action]['dose']

        self.current_step += 1

        if self.current_step < len(self.current_infusions):
            next_state = self.get_state()
            reward = self.calculate_reward(next_state)
            done = False
        else:
            next_state = np.array([-1, -1, -1])
            reward = 0
            done = True

        return next_state, reward, done, {}

    def calculate_reward(self, state):
        heart_rate, sao2, respiration = state
        reward = 0
        if heart_rate != -1 and heart_rate < 100:
            reward += 1
        if sao2 != -1 and sao2 > 90:
            reward += 1
        if respiration != -1 and respiration < 20:
            reward += 1
        return reward

    def render(self, mode='human'):
        pass

# 数据文件路径
patient_data_path = '/Users/tony.ren/Desktop/patient1.csv '
infusion_data_path = '/Users/tony.ren/Desktop/infusion.csv '
vital_data_path = '/Users/tony.ren/Desktop/vitalperiodic3.csv '

# 创建环境
env = SepsisTreatmentEnv(patient_data_path, infusion_data_path, vital_data_path)

# 初始化 DQN 模型
model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001, buffer_size=10000, exploration_fraction=0.1, exploration_final_eps=0.01, target_update_interval=100)

# 创建评估回调
eval_callback = EvalCallback(env, best_model_save_path='./logs/best_model.zip', log_path='./logs/results', eval_freq=500, deterministic=True, render=False)

# 训练模型
model.learn(total_timesteps=10000, callback=eval_callback)

# 测试模型
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()