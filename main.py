from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot
import torch.multiprocessing as mp

# intervals = {'training': ('2008-12-31', '2015-09-30'),
#              'validation': ('2015-10-01', '2015-12-31'),
#              'testing': ('2016-01-04', '2021-08-04')}

intervals = {'training': ('2008-03-19', '2017-09-05'),
             'validation': ('2017-09-06', '2019-09-17'),
             'testing': ('2019-09-18', '2021-09-27')}


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    ddpg = DDPG(intervals)
    ddpg.train()
    ddpg.test()

    # ppo = PPO(intervals)
    # ppo.train(verbose=True)
    # ppo.test()

    # a2c = A2C(intervals, n_agents=1)
    # a2c.train(verbose=True)
    # a2c.test()


if __name__ == '__main__':
    main()
