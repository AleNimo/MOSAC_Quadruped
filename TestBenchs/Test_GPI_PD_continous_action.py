import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
import numpy as np

from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction

GAMMA = 0.99

env = mo_gym.make("mo-halfcheetah-v4")

env = MORecordEpisodeStatistics(env, gamma=GAMMA)  # wrapper for recording statistics

eval_env = mo_gym.make("mo-halfcheetah-v4")

agent = GPIPDContinuousAction(env, gamma = GAMMA, batch_size = 1000, device = "cuda", seed=1)

agent.train(total_timesteps = 100000, eval_env = eval_env, ref_point = np.array([-100, -100]), checkpoints = True)  #Approximately 9/9.5hs of training

