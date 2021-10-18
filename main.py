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

    for i in range(5):
        if os.path.isfile(f'plots/ddpg/{i}2_testing.png'):
            print('already done!')
            continue
        ddpg = DDPG(state_type='indicators', djia_year=2019, repeat=i)
        ddpg.train()
        ddpg.test()

        if os.path.isfile(f'plots/ppo/{i}2_testing.png'):
            print('already done!')
            continue
        ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
        ppo.train()
        ppo.test()

        if os.path.isfile(f'plots/a2c/{i}2_testing.png'):
            print('already done!')
            continue
        a2c = A2C(n_agents=8, state_type='indicators', djia_year=2019, repeat=i)
        a2c.train()
        a2c.test()


if __name__ == '__main__':
    main()
