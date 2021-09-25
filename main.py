from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot

intervals = {'training': ('2009-01-02', '2015-09-30'),
             'validation': ('2015-10-01', '2015-12-31'),
             'testing': ('2016-01-04', '2021-08-04')}


def main():

    plot.initialize()

    ddpg = DDPG(intervals)
    ddpg.train(verbose=True)
    ddpg.test()

    ppo = PPO(intervals)
    ppo.train(verbose=True)
    ppo.test()

    a2c = A2C(intervals, n_agents=4)
    a2c.train(verbose=True)
    a2c.test()


if __name__ == '__main__':
    main()
