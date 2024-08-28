import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
import numpy as np

from morl_baselines.multi_policy.capql.capql import CAPQL

GAMMA = 0.99

env = mo_gym.make("mo-walker2d-v4", render_mode="human")

env = MORecordEpisodeStatistics(env, gamma=GAMMA)  # wrapper for recording statistics

eval_env = mo_gym.make("mo-walker2d-v4", render_mode="human")

agent = CAPQL(env, gamma = GAMMA, batch_size = 1000, device = "cuda", seed=1)

agent.train(total_timesteps = 1000000, eval_env = eval_env, ref_point = np.array([-100, -100]), eval_freq = 2000000, checkpoints = True) #Approximately 8:30/9hs of training

