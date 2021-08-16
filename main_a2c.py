from env.environment import PortfolioEnv
import numpy as np
import torch as T
from agents.agent_a2c import ActorCritic
from agents.agent_a2c import Agent
from plot import add_curve, save_plot

N_AGENTS = 4
GAMMA = 0.99
T_MAX = 5


def main():
    figure_file = 'plots/a2c.png'

    env = PortfolioEnv()
    djia_history = env.get_djia_history()
    add_curve(djia_history/djia_history[0], 'DJIA')

    global_actor_critic = ActorCritic(input_dims=env.state_shape(), n_actions=env.n_actions(), fc1_dims=128)
    optimizer = T.optim.Adam(global_actor_critic.parameters())

    workers = [Agent(global_actor_critic,
                     input_dims=env.state_shape(),
                     n_actions=env.n_actions(),
                     gamma=GAMMA,
                     name=i,
                     t_max=T_MAX,
                     layer1_size=128) for i in range(N_AGENTS)]

    while Agent.n_dones < N_AGENTS:
        [w.iterate() for w in workers]
        if Agent.n_gradients == N_AGENTS:
            gradients = np.array([w.get_gradient() for w in workers], dtype=object)
            mean_gradient = np.mean(gradients, axis=0)
            for grad, global_param in zip(
                    mean_gradient,
                    global_actor_critic.parameters()):
                global_param._grad = grad
            optimizer.step()
            [w.resume() for w in workers]
            # print("------ global network updated ------")

    save_plot(figure_file)


if __name__ == '__main__':
    main()
