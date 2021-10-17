import warnings
warnings.filterwarnings('ignore')

from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot
import torch.multiprocessing as mp


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    for year in [2012, 2019]:
        for i in range(5):
            ddpg = DDPG(state_type='indicators', djia_year=year, repeat=i)
            ddpg.train()
            ddpg.test()

            ppo = PPO(state_type='indicators', djia_year=year, repeat=i)
            ppo.train()
            ppo.test()

            # a2c = A2C(n_agents=8, state_type='indicators', djia_year=year, repeat=i)
            # a2c.train()
            # a2c.test()


if __name__ == '__main__':
    main()
