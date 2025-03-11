import gym
import compiler_gym
import random
import os
from compiler_gym.envs import CompilerEnv
import math
import optuna

#stable baselines
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

f1data = []
f2data = []
f3data = []
f4data = []

class CompilerGymWrapperNormal(gym.Wrapper):

    def __init__(self, env: CompilerEnv):
        super(CompilerGymWrapperNormal, self).__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_count = 0

    def reset(self):
        self.action_count = 0
        return self.env.reset()

    def step(self, action):
        action = int(action)
        self.action_count += 1
        observation, reward, done, info = self.env.step(action)
        if (self.action_count > 40):
            done = True
        return observation, reward, done, info


def Optimize(trial):
    env = gym.make("llvm-autophase-ic-v0",benchmark="cbench-v1/bzip2",observation_space="Autophase")
    wrapped_env_normal = CompilerGymWrapperNormal(env)

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    n_steps = trial.suggest_int("n_steps", 128, 2048, step=128)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)

    net_arch = [dict(pi=[256,256,256,256],vf=[256,256,256,256])]


    model = PPO('MlpPolicy', wrapped_env_normal,learning_rate=learning_rate, gae_lambda=gae_lambda, gamma=gamma, n_steps=n_steps, verbose=1, policy_kwargs={"net_arch": net_arch})
    model.learn(total_timesteps=10000)
    mean_reward, _ = evaluate_policy(model, wrapped_env_normal, n_eval_episodes=30)
    
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(Optimize, n_trials=20, n_jobs= 4)


print("Best HyperParameters: ", study.best_params)