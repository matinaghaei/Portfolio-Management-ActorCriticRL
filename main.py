from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot
import torch.multiprocessing as mp
import os


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    layer_size = [(1024, 1024, 1024), (768, 768, 768), (512, 512, 512), (1024, 1024, None), (768, 768, None), (512, 512, None), (400, 300, None), (128, None, None)]
    state_type = ['indicators', 'only prices']
    djia_year = [2012, 2019]
    bn_drop = ['only bn', 'bn drop', 'without noise', 'tanh', 'without noise tanh']
    action_input_layer = ['nothing', 'bn drop']
    action_interpret = ['portfolio', 'transactions']

    range1 = [6]
    range2 = [0]
    range3 = [1]
    range4 = [4]
    range5 = [0]
    range6 = [0]
    

    for ls in range1:
        for st in range2:
            for dy in range3:
                for bd in range4:
                    for ail in range5:
                        for ai in range6:
                            for i in range(10):
                                if os.path.isfile(f'plots/ddpg/{layer_size[ls][0]}_{layer_size[ls][1]}_{layer_size[ls][2]}_{state_type[st]}_{djia_year[dy]}_{bn_drop[bd]}_{action_input_layer[ail]}_{action_interpret[ai]}/{i}2_testing.png'):
                                    print('already done!')
                                    continue
                                ddpg = DDPG(layer1_size=layer_size[ls][0], layer2_size=layer_size[ls][1],
                                            state_type=state_type[st], djia_year=djia_year[dy], repeat=i, 
                                            bn_drop=bn_drop[bd], action_input_layer=action_input_layer[ail], action_interpret=action_interpret[ai])
                                ddpg.train()
                                ddpg.test()

    # ppo = PPO()
    # ppo.train(verbose=True)
    # ppo.test()

    # a2c = A2C(n_agents=1)
    # a2c.train(verbose=True)
    # a2c.test()


if __name__ == '__main__':
    main()
