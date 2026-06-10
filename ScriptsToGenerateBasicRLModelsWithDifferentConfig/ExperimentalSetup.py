import gym
import compiler_gym
import random
import os
from compiler_gym.envs import CompilerEnv
import math

#stable baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

f1data = []
f2data = []
f3data = []
f4data = []

def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def SaveEvalData(model, env, fileName, steps, oginstrcount, benchmark, obsspace, originalruntime):

    episodes = 100

    running_inst_cnt = 0
    best_inst_cnt = 100000000
    worst_inst_cnt = 0

    running_rewards = 0
    best_rewards = 100000000
    worst_rewards = 0

    for i in range (1, episodes + 1 ):
        obs = env.reset()
        
        done = False
        score = 0
        actions = []
        best_actions = []
        
        while not done:
            action, _ = model.predict(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            score += reward
        
        try:
            instCnt = env.observation["IrInstructionCount"]
            running_inst_cnt += instCnt
            running_rewards += score

            if (score < worst_rewards):
                worst_rewards = score
            if (score > best_rewards):
                best_rewards = score

            if (instCnt > worst_inst_cnt):
                worst_inst_cnt = instCnt
            if (instCnt < best_inst_cnt):
                best_inst_cnt = instCnt
                best_actions = actions

            
            f1data.append(fileName + ','+ str(steps) + ','+ str(env.observation["IrInstructionCount"]) + ','+ str(truncate(score,4)) + ','+ str(truncate(int(oginstrcount) / instCnt,2))+',' + str(env.observation["Runtime"][0]) + ','+ str(truncate(originalruntime[0],2))+'\n')
        except compiler_gym.errors.ServiceError as e:
            print("Session not active:", e)


    f2data.append(fileName + ','+ str(steps) + ','+ str(best_inst_cnt) + ','+ str(worst_inst_cnt) + ','+str(truncate(running_inst_cnt / episodes,2))+'\n')


    f3data.append(fileName + ','+ str(steps) +','+ str(truncate(score,4)) + ','+ str(truncate(score,4)) + ','+ str(truncate(running_rewards / episodes,2))+'\n')



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
        if (self.action_count >= 40): #restrict the actions to 40
            done = True
        return observation, reward, done, info
    
def CreateModelAndLearn(testname, algorithm, maxsteps, pbenchmark, observataionSpace, stepvalue, layernodes, tuned):

    env = gym.make("llvm-autophase-ic-v0",benchmark=pbenchmark,observation_space=observataionSpace)
    wrapped_env_normal = CompilerGymWrapperNormal(env)

    wrapped_env_normal.reset()
    log_path = os.path.join('Training_Comparisson_Algo','Logs_'+algorithm)
    net_arch = [dict(pi=[layernodes,layernodes,layernodes,layernodes],vf=[layernodes,layernodes,layernodes,layernodes])]


    originalInstCount = str(env.observation["IrInstructionCount"])
    runtime = str(env.observation["Runtime"])

    f1 = open(algorithm+"_details.txt", "a")
    f1.writelines(originalInstCount+'\n')
    f1.close()

    if (algorithm == 'PPO'):
        if (layernodes > 0):
            if (tuned):
                print("runing tuned layered PPO")
                model = PPO('MlpPolicy', wrapped_env_normal, verbose=1, policy_kwargs={"net_arch": net_arch},
                    learning_rate= 0.000215, gae_lambda= 0.886880,gamma= 0.904995,n_steps= 768)
            else:
                print("runing layered PPO")
                model = PPO('MlpPolicy', wrapped_env_normal, verbose=1, policy_kwargs={"net_arch": net_arch})
        else:
            if (tuned):
                model = PPO('MlpPolicy', wrapped_env_normal, verbose=1,
                    learning_rate= 0.000215, gae_lambda= 0.886880,gamma= 0.904995,n_steps= 768)
                print("runing tuned plain PPO")
            else:
                model = PPO('MlpPolicy', wrapped_env_normal, verbose=1)
                print("runing just plain PPO")

    elif (algorithm == 'DQN'):
        model = DQN('MlpPolicy', wrapped_env_normal, verbose=1, learning_rate=0.00012,buffer_size=200000,batch_size=64,
                    gamma= 0.884339, exploration_fraction=0.2367176, exploration_final_eps=0.06087314, target_update_interval= 1000, policy_kwargs={"net_arch": [256, 256]} )
    if (algorithm == 'A2C'):
        if (layernodes > 0):
            if (tuned):
                model = A2C('MlpPolicy', wrapped_env_normal, verbose=1, policy_kwargs={"net_arch": net_arch},
                    learning_rate= 0.000215, gae_lambda= 0.886880,gamma= 0.904995,n_steps= 768)
                print("runing tuned layered A2C")
            else:
                model = A2C('MlpPolicy', wrapped_env_normal, verbose=1, policy_kwargs={"net_arch": net_arch})
                print("runing layered A2C")
        else:
            if (tuned):
                model = A2C('MlpPolicy', wrapped_env_normal, verbose=1,
                    learning_rate= 0.000215, gae_lambda= 0.886880,gamma= 0.904995,n_steps= 768)
                print("runing tuned plain A2C")
            else:
                model = A2C('MlpPolicy', wrapped_env_normal, verbose=1)
                print("runing plain A2C")

    timestep_count = stepvalue

    while timestep_count <= maxsteps:
        wrapped_env_normal.reset()
        print("Timesteps:"+str(timestep_count))
        model.learn(total_timesteps = stepvalue)

        if (timestep_count == maxsteps):
            Model_Path = os.path.join('Training',testname+'_'+algorithm+'_Model_Experiment_'+str(timestep_count))
            model.save(Model_Path)

        SaveEvalData(model, wrapped_env_normal, testname, timestep_count,originalInstCount,pbenchmark,observataionSpace,runtime)
        timestep_count += stepvalue


def FinalFileWriteWithCreateModelAndLearn(testname, algorithm, maxsteps, pbenchmark, observataionSpace, stepvalue, layernodes, tuned):

    CreateModelAndLearn(testname,algorithm, maxsteps, pbenchmark, observataionSpace, stepvalue, layernodes, tuned)
    bench = pbenchmark.replace("/","_")

    prefix = ""
    f1_name = prefix + bench+'_'+algorithm+'_'+observataionSpace+"_fulldatapoints.csv"
    f2_name = prefix+ bench+'_'+algorithm+'_'+observataionSpace+"_instr_count.csv"
    f3_name = prefix+ bench+'_'+algorithm+'_'+observataionSpace+"_reward_details.csv"

    f1 = open(f1_name, "a")
    f2 = open(f2_name, "a")
    f3 = open(f3_name, "a")

    for i in f1data:
        f1.writelines(i)
    
    for i in f2data:
        f2.writelines(i)

    for i in f3data:
        f3.writelines(i)

    f1.close()
    f2.close()
    f3.close()

    
#This method trains a model for given steps and evaluate from a defined steps count and save the csv file with data
#This model will Save the model after training
#change here and run for different configurations
#parameters are: outputfilename, algorithm, totalsteps, benchmark, observationspace, evaluateStepCount, nodes for a layer, UseTunedHyperParameters
#algorithm parameter could be PPO, DQN and A2C
#observation space could be Autophase or InstCount
#nodes for a layer is generaly from 128,256 or 512
#UseTuneHyperParameters: True or False
FinalFileWriteWithCreateModelAndLearn('PPO_20000_bzip2__instcount_128', 'PPO', 20000,"cbench-v1/bzip2","InstCount",1000,128,False)
