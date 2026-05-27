import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import time
import STP
import TD3
import warnings
plt.switch_backend('agg')
warnings.filterwarnings("ignore",category=DeprecationWarning)

def eval_policy(policy, env_name, seed):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(5):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    return avg_reward / 5

def train(Env, Seed, j, ip, k):
	start_time = time.time()

	evaluations = []
	file_name = "%s_%s_%s_%s" % (Env, str(Seed), str(ip), str(k))

	env = gym.make(Env)
	env.seed(Seed)
	torch.manual_seed(Seed)
	np.random.seed(Seed)
	env.action_space.seed(Seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	policy = TD3.policy(state_dim, action_dim, max_action)
	replay_buffer = STP.PrioritizedReplayBuffer(state_dim, action_dim,ip,k)

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(1e6)+1):
		episode_timesteps += 1

		if t < 25e3:
			action = env.action_space.sample()
		else:
			noise = np.random.normal(0, max_action * 0.1, size=action_dim)
			action = (policy.select_action(np.array(state)) + noise).clip(-max_action, max_action)

		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		replay_buffer.add(state, action, next_state, reward, done_bool)

		if t >= 25e3:
			policy.train(replay_buffer, 256)

		state = next_state
		episode_reward += reward
		if done:
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		if t % 5e3 == 0 and t > 0:
			R = eval_policy(policy, Env, Seed)
			evaluations.append(R)
			if t % 1e4 == 0:
				print(f"Step: {t}  Reward: {int(R)}")
			if t % int(1e5) == 0:
				print(f"--------  {round((time.time() - start_time) / 60., 1)} min  --------")
			if t == int(1e6):
				np.save(f"./results/{file_name}", evaluations)
				plt.figure(j)
				plt.plot(evaluations, 'b')
				plt.savefig("./results/" + file_name + "-" + str(round(np.max(evaluations))) + '.png')


if __name__ == "__main__":

	EnvList = ["HalfCheetah-v4", "Ant-v4", "Humanoid-v4", "Walker2d-v4", "Hopper-v4"]  # "HalfCheetah", "Ant-v4",
	SeedList = np.array([10,11,12]) #

	j = int(0)
	for seq in range(SeedList.shape[0]):
		for enq in range(len(EnvList)):
			Env = EnvList[enq]
			j = j + 1
			ip = 0.2
			k = 10
			Seed = int(SeedList[seq])
			print(f"---- Env: {Env}, Seed: {Seed} ----")
			train(Env, int(Seed), j,ip,k)
