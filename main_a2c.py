import numpy as np
import torch as T
from agents.agent_a2c import ActorCritic
from agents.agent_a2c import Agent

N_AGENTS = 8
GAMMA = 0.99
T_MAX = 5


def main():

    n_agents = [1]
    fc_sizes = [128]
    entropies = [0]
    for a in n_agents:
        for fc in fc_sizes:
            for e in entropies:
                Agent.results = []
                for i in range(20):
                    Agent.n_dones = 0
                    Agent.n_gradients = 0
                    Agent.score_history = []

                    global_actor_critic = ActorCritic(input_dims=(61,), n_actions=30, fc1_dims=fc, fc2_dims=fc)
                    optimizer = T.optim.Adam(global_actor_critic.parameters())

                    workers = [Agent(global_actor_critic,
                                     input_dims=(61,),
                                     n_actions=30,
                                     gamma=GAMMA,
                                     name=i,
                                     t_max=T_MAX,
                                     layer1_size=fc,
                                     layer2_size=fc,
                                     entropy_coef=e) for i in range(a)]

                    while Agent.n_dones < a:
                        [w.iterate() for w in workers]
                        if Agent.n_gradients == a:
                            gradients = np.array([w.get_gradient() for w in workers], dtype=object)
                            mean_gradient = np.mean(gradients, axis=0)
                            for grad, global_param in zip(
                                    mean_gradient,
                                    global_actor_critic.parameters()):
                                global_param._grad = grad
                            optimizer.step()
                            [w.resume() for w in workers]
                            # print("------ global network updated ------")

                print(f"average: {sum(Agent.results) / len(Agent.results)}\n")


if __name__ == '__main__':
    main()
