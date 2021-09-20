from env.environment import PortfolioEnv
from agents.agent_a2c import ActorCritic, Agent
from plot import add_curve, save_plot
from multiprocessing import Pipe, Lock
import time

N_AGENTS = 4


def main():
    env = PortfolioEnv()
    network = ActorCritic(input_dims=env.state_shape(), n_actions=env.n_actions(), gamma=0.99, fc1_dims=128, lr=1e-3)

    pipes = [Pipe() for i in range(N_AGENTS)]
    local_conns, remote_conns = list(zip(*pipes))
    lock = Lock()
    workers = [Agent(network,
                     lock,
                     conn=remote_conns[i],
                     name=f'w{i}',
                     t_max=5) for i in range(N_AGENTS)]
    [w.start() for w in workers]

    while not network.done:
        losses = []
        for c in local_conns:
            network.set_memory(*c.recv())
            losses.append(network.calc_loss())
        total_loss = sum(losses)
        network.zero_grad()
        total_loss.backward()
        network.optimizer.step()
        [c.send('resume') for c in local_conns]


if __name__ == '__main__':
    main()
