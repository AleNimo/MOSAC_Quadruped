import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
import numpy as np

from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL

GAMMA = 0.99

eval_env = mo_gym.make("mo-halfcheetah-v4", render_mode="human")

agent = PGMORL(env_id="mo-halfcheetah-v4", origin = np.array([-100, -100]), gamma = GAMMA, log = True, device = "cuda")

agent.train(total_timesteps = 3000000, eval_env = eval_env, ref_point = np.array([-100, -100]))