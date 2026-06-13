import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import argparse
import os
import time
import warnings

import STP
import TD3

plt.switch_backend("agg")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

def eval_policy(policy, env_name, seed, eval_episodes=5):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="HalfCheetah-v4")              # OpenAI gym environment name
	parser.add_argument("--seed", default=10, type=int)                 # Sets Gym, PyTorch and Numpy seeds

	parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)           # How often we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)          # Batch size

	parser.add_argument("--alpha", default=0.2, type=float)             # Spatial priority clipping α
	parser.add_argument("--k", default=10, type=int)                    # Number of initial groups k
	parser.add_argument("--lambda_", default=0.5, type=float)            # Trade-off factor λ 

	parser.add_argument("--eval_episodes", default=5, type=int)         # Evaluation episodes
	parser.add_argument("--save_plot", action="store_true")             # Save reward curve

	args = parser.parse_args()

	file_name = f"{args.env}_{args.seed}_{args.alpha}_{args.k}_{args.lambda_}"

	print("---------------------------------------")
	print(f"Env: {args.env}, Seed: {args.seed}, alpha: {args.alpha}, k: {args.k}, lambda: {args.lambda_}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	start_time = time.time()

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	# Initialize policy
	policy = TD3.policy(state_dim, action_dim, max_action)

	# Initialize STP replay buffer
	replay_buffer = STP.PrioritizedReplayBuffer(
		state_dim,
		action_dim,
		args.alpha,
		args.k,
		args.lambda_
	)

	evaluations = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps) + 1):

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done:
			print(
				f"Total T: {t + 1} "
				f"Episode Num: {episode_num + 1} "
				f"Episode T: {episode_timesteps} "
				f"Reward: {episode_reward:.3f}"
			)

			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if t % args.eval_freq == 0 and t > 0:
			R = eval_policy(
				policy,
				args.env,
				args.seed,
				eval_episodes=args.eval_episodes
			)

			evaluations.append(R)

			np.save(f"./results/{file_name}", evaluations)

			if t % int(1e4) == 0:
				print(f"Step: {t}  Reward: {int(R)}")

			if t % int(1e5) == 0:
				print(f"--------  {round((time.time() - start_time) / 60., 1)} min  --------")

			if args.save_plot:
				plt.figure()
				plt.plot(evaluations, "b")
				plt.xlabel("Evaluation")
				plt.ylabel("Average Reward")
				plt.title(file_name)
				plt.savefig(
					"./results/" + file_name + "-" + str(round(np.max(evaluations))) + ".png"
				)
				plt.close()