from algorithms.ddpg.ddpg import DDPG
from algorithms.a2c.a2c import A2C
from algorithms.ppo.ppo import PPO
import plot
import torch.multiprocessing as mp


def main():

    plot.initialize()
    mp.set_start_method('spawn')

    train_eval = ['old', 'new']
    bn_drop = ['only bn', 'only drop', 'bn drop']
    action_input_layer = ['nothing', 'bn drop']
    state_activation = [False, True]
    action_activation = [True, False]
    
    for te in train_eval:
        for bd in bn_drop:
            for ail in action_input_layer:
                for sa in state_activation:
                    for aa in action_activation:
                        for i in range(5):
                            ddpg = DDPG(layer1_size=400, layer2_size=300, layer3_size=None,
                                        action_interpret='portfolio', state_type='only prices', djia_year=2019,
                                        train_eval=te, bn_drop=bd, action_input_layer=ail,
                                        state_activation=sa, action_activation=aa, repeat=i)
                            ddpg.train()
                            ddpg.test(verbose=False)

    # ppo = PPO()
    # ppo.train(verbose=True)
    # ppo.test()

    # a2c = A2C(n_agents=1)
    # a2c.train(verbose=True)
    # a2c.test()


if __name__ == '__main__':
    main()
