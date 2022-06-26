import torch
from wrappers.wrappers import Wrappers
from models.DQN import DQN
from buffer.experienceBuffer import ExperienceBuffer
from agents.DDQN_agent import Agent
import torch.optim as optim
import numpy as np

#Initial values
REPLAY_SIZE = 10000
BATCH_SIZE = 50
NET_UPDATE = 1000
REPLAY_BATCH_SIZE = 10000
EPSILON_0, EPSILON_F, EPSILON_DECAY = 1.0, 0.01, 10**4
NAME, DEFAULT_ENV_NAME='Breakout', 'ALE/Breakout-v5'
MAX_EPISODES, MAX_STEPS = 100000, 100000

if __name__ == "__main__":
    epsilon = EPSILON_0
    device = torch.device("cuda" if torch.cuda else "cpu")
    env = Wrappers.make_env(DEFAULT_ENV_NAME)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env,net,target_net, buffer)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    episode_rewards = []
    frame = 0
    game=0
    record=0

    for episode in range(MAX_EPISODES):
        game+=1
        score=0    
        for step in range(MAX_STEPS):
            frame += 1
            epsilon = max(EPSILON_F, EPSILON_0 - frame / EPSILON_DECAY)

            action = agent.get_action(net,device,epsilon)
            reward,done=agent.step(action)

            if reward is not None:
                episode_rewards.append(reward)
                score=reward
                mean_score = np.mean(episode_rewards[-50:])

                if score>record:
                    record=score
                    print("Game: {}, New best reward {}, best reward {}, mean score {}, saving model...".format(episode, score, record,mean_score))
                    torch.save(net.state_dict(), './learnedModels/'+NAME+'_ddqn_model.pt')
                    torch.save(target_net.state_dict(), './learnedModels/'+NAME+'_ddqn_target_model.pt')
                else:
                    print("Game: {}, reward {}, best reward {}, mean score {}.".format(episode, score, record,mean_score))
            
            if len(buffer) >= REPLAY_BATCH_SIZE:
                if frame % NET_UPDATE == 0:
                    target_net.load_state_dict(net.state_dict())

                agent.update(optimizer,buffer,BATCH_SIZE,device)

            if done:
                break