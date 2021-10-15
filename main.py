from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot
import torch.multiprocessing as mp
import os


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    layer_size = [(1024, 1024, 1024), (768, 768, 768), (512, 512, 512), (1024, 1024, None), (768, 768, None), (512, 512, None), (400, 300, None), (256, 256, None)]
    state_type = ['indicators', 'only prices']
    djia_year = [2012, 2019]
    entropy = [0, 1e-2, 1e-4]

    range1 = [6]
    range2 = [0]
    range3 = [1]
    range4 = [0]

    for ls in range1:
        for st in range2:
            for dy in range3:
                for ent in range4:
                    for i in range(10):
                        if os.path.isfile(f'plots/ddpg/{layer_size[ls][0]}_{layer_size[ls][1]}_{layer_size[ls][2]}_{state_type[st]}_{djia_year[dy]}/{i}2_testing.png'):
                            print('already done!')
                            continue
                        ddpg = DDPG(layer1_size=layer_size[ls][0], layer2_size=layer_size[ls][1],
                                    state_type=state_type[st], djia_year=djia_year[dy], repeat=i)
                        ddpg.train()
                        ddpg.test()

                        # if os.path.isfile(f'plots/ppo/{layer_size[ls][0]}_{layer_size[ls][1]}_{layer_size[ls][2]}_{state_type[st]}_{djia_year[dy]}_{entropy[ent]}/{i}2_testing.png'):
                        #     print('already done!')
                        #     continue
                        # ppo = PPO(layer1_size=layer_size[ls][0], layer2_size=layer_size[ls][1],
                        #             state_type=state_type[st], djia_year=djia_year[dy], repeat=i, entropy=entropy[ent])
                        # ppo.train()
                        # ppo.test()

                        # if os.path.isfile(f'plots/a2c/{layer_size[ls][0]}_{layer_size[ls][1]}_{layer_size[ls][2]}_{state_type[st]}_{djia_year[dy]}_{entropy[ent]}/{i}2_testing.png'):
                        #     print('already done!')
                        #     continue
                        # a2c = A2C(n_agents=8, layer1_size=layer_size[ls][0], layer2_size=layer_size[ls][1],
                        #           state_type=state_type[st], djia_year=djia_year[dy], repeat=i, entropy=entropy[ent])
                        # a2c.train()
                        # a2c.test()


if __name__ == '__main__':
    main()
