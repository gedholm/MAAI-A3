import gym
from policy_gradient import PolicyGradient
from functions_actions import *
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('BipedalWalker-v2')
env = env.unwrapped

env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

RENDER_ENV = False
EPISODES = 50000
rewards = []
RENDER_REWARD_MIN = 50
MAX_FRAMES = 1800
N_avg = 100

if __name__ == "__main__":


    # Load checkpoint
    load_path = None #"output/weights/CartPole-v0.ckpt"
    save_path = None #"output/weights/CartPole-v0-temp.ckpt"

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = 9,
        learning_rate=0.01,
        reward_decay=0.995,
        load_path=load_path,
        save_path=save_path
    )

    past_n_rews = []

    for episode in range(EPISODES+1):

        observation = env.reset()
        episode_reward = 0
        frame_counter = 0
        while True:
            if RENDER_ENV: 
                print("rendering while training")
                PG.run_simulation(MAX_FRAMES, env, True)
            # 1. Choose an action based on observation
            a = PG.choose_action(observation)
            action = make_action_int(a)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)
            frame_counter += 1
            if frame_counter > MAX_FRAMES:
                done = True
            # 3. Store transition for training
            PG.store_transition(observation, a, reward)
            observation = observation_

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                if episode_rewards_sum > RENDER_REWARD_MIN:
                    PG.run_simulation(MAX_FRAMES, env, True)
                rewards.append(episode_rewards_sum)
                past_n_rews.append(episode_rewards_sum)
                if(len(past_n_rews)>N_avg):
                    past_n_rews.pop(0)
                avg_last_n = sum(past_n_rews)/len(past_n_rews)
                max_reward_so_far = np.amax(rewards)
                RENDER_ENV = False
                if episode%100 == 0:
                    RENDER_ENV = True
                if episode%100 == 0:
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Avg last", len(past_n_rews), ": ", avg_last_n)
                    print("Reward: ", episode_rewards_sum)
                    print("Max reward so far: ", max_reward_so_far)
                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn()
                # Render env if we get to rewards minimum
                if episode_rewards_sum > RENDER_REWARD_MIN: RENDER_ENV = True

                break

            # Save new observation
            observation = observation_
    
    final_returns = []
    print("Final trials")
    for i in range(0, 50):
        final_returns.append(PG.run_simulation(MAX_FRAMES, env, False))
    best_final_return = max(final_returns)
    avg_final_return = sum(final_returns)/len(final_returns)
    print("Best final return: ", best_final_return)
    print("Average final return: ", avg_final_return)
