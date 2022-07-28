import torch
from wrappers.wrappers import Wrappers
from models.A3C import A3C
from trainA3C import train
from torch.multiprocessing import Process
from main_thread import *

NAME, DEFAULT_ENV_NAME='Breakout', 'ALE/Breakout-v5'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    torch.manual_seed(8967)
    env = Wrappers.make_env(DEFAULT_ENV_NAME)

    shared_model = A3C(env.observation_space.shape, env.action_space.n).to(device)

    shared_model.share_memory()

    processes = []
    for rank in range(4):
        p = Process(target=train, args = (rank, shared_model))
        p.start()
        processes.append(p)

    main_thread(env, shared_model)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()