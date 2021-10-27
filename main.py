import warnings
warnings.filterwarnings('ignore')

from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot
import torch.multiprocessing as mp
import os


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    for i in range(50):
        print(f"---------- round {i} ----------")

        if not os.path.isfile(f'plots/ddpg/{i}2_testing.png'):
            ddpg = DDPG(state_type='indicators', djia_year=2019, repeat=i)
            ddpg.train()
            ddpg.test()

        if not os.path.isfile(f'plots/ppo/{i}2_testing.png'):
            ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
            ppo.train()
            ppo.test()

        if not os.path.isfile(f'plots/a2c/{i}2_testing.png'):
            a2c = A2C(n_agents=8, state_type='indicators', djia_year=2019, repeat=i)
            a2c.train()
            a2c.test()


if __name__ == '__main__':
    main()
