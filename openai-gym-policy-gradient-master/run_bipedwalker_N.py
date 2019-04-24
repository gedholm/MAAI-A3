""" This script creates an instance of the PolicyGradientN class,
    trains it on the """


import gym
from policy_gradient_n import PolicyGradientN
from functions_actions_N import *
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('BipedalWalker-v2')
env = env.unwrapped

env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

n_x = env.observation_space.shape[0] - 10
n_y = 3
RENDER_ENV = False
LR = 0.01
GAMMA = 0.98
EPISODES = 9000
rewards = []
RENDER_REWARD_MIN = 30
N_HEURISTIC = 10
N_avg = 100
RENDER_INTERVAL = 100
CONVERGENCE_LIMIT = 0
CONVERGENCE_COUNTER_LIMIT = 5
EPS = 1
best_PGs = []
best_results = []

if __name__ == "__main__":


    # Load checkpoint
    load_path = None #"output/weights/CartPole-v0.ckpt"
    save_path = None #"output/weights/CartPole-v0-temp.ckpt"
    PGiteration = 1
    PG = PolicyGradientN(
        n_x = n_x, #remove lidar,
        n_y = n_y,
        learning_rate=LR,
        reward_decay=GAMMA,
        load_path=load_path,
        save_path=save_path,
        N=4,
        iteration=PGiteration
    )

    past_n_rews = []
    convergence_counter = 0
    prev_avg = 0
    avg_last_n = [0]
    improvement_counter = 0
    max_reward_so_far = -3000
    for episode in range(EPISODES+1):
        observation = env.reset()[0:14]
        episode_reward = 0
        frame_counter = 0
        max_frames = 450#random.randrange(200, 500)
        done = False
        if episode%N_HEURISTIC == 0:
            heuristic_walk(observation, env, PG, episode)
        while True:
            if RENDER_ENV: env.render()
            # 1. Choose an action based on observation
            one_hot_1, one_hot_2, one_hot_3, one_hot_4 = PG.choose_action(observation)
            a1 = make_action_from_one(one_hot_1, 0)
            a2 = make_action_from_one(one_hot_2, 1)
            a3 = make_action_from_one(one_hot_3, 2)
            a4 = make_action_from_one(one_hot_4, 3)
            action = sum_actions(a1, a2, a3, a4)
            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)
            observation_ = observation_[0:14]
            frame_counter += 1
            if frame_counter > max_frames:
                done = True
            # 3. Store transition for training
            PG.store_transition(observation, one_hot_1, one_hot_2, one_hot_3, one_hot_4, reward)
            observation = observation_

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                if RENDER_ENV: print("Reward in render:", episode_rewards_sum, " Frames: ", frame_counter)
                if episode_rewards_sum > RENDER_REWARD_MIN:
                    print("good one")
                    PG.run_simulation(3000, env, True)
                    RENDER_REWARD_MIN = episode_rewards_sum
                    best_PGs.append(PG)
                    # PGiteration += 1
                    # improvement_counter = 0
                    # PG = PolicyGradientN(
                    # n_x = n_x,
                    # n_y = n_y,
                    # learning_rate=LR,
                    # reward_decay=GAMMA,
                    # load_path=load_path,
                    # save_path=save_path,
                    # N=4, 
                    # iteration=PGiteration
                    # )
                rewards.append(episode_rewards_sum)
                past_n_rews.append(episode_rewards_sum)
                if(len(past_n_rews)>N_avg):
                    past_n_rews.pop(0)
                max_reward_so_far = np.amax(rewards)
                RENDER_ENV = False
                if episode%RENDER_INTERVAL == 0:
                    RENDER_ENV = True
                if episode%100 == 0:
                    avg_last_n = sum(past_n_rews)/len(past_n_rews)
                    if abs(avg_last_n - prev_avg) < EPS: convergence_counter += 1
                    else: convergence_counter = 0
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Avg last", len(past_n_rews), ": ", avg_last_n)
                    print("Best last", len(past_n_rews), ": ", max(past_n_rews))
                    print("Reward: ", episode_rewards_sum)
                    print("Max reward so far: ", max_reward_so_far)
                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn()
                # Render env if we get to rewards minimum
                if episode_rewards_sum > RENDER_REWARD_MIN: RENDER_ENV = True
                prev_avg = avg_last_n
                if convergence_counter>CONVERGENCE_COUNTER_LIMIT and avg_last_n < CONVERGENCE_LIMIT:
                    convergence_counter = 0
                    PGiteration += 1
                    LR *= 5
                    PG = PolicyGradientN(
                        n_x = n_x,
                        n_y = n_y,
                        learning_rate=LR,
                        reward_decay=GAMMA,
                        load_path=load_path,
                        save_path=save_path,
                        N=4, 
                        iteration=PGiteration
                    )
                    print("reinitialize")
                break
    if len(best_PGs) == 0: best_PGs.append(PG)
    for PG_i in best_PGs:
        res_i = run_trials(PG_i, env)
        best_results.append(res_i)
    print(best_results)