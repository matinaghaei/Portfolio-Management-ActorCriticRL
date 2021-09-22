import time
from env.environment import PortfolioEnv
from algorithms.a2c.agent import ActorCritic, Agent
from multiprocessing import Pipe, Lock


class A2C:

    def __init__(self, n_agents, load=False, alpha=1e-3, gamma=0.99, layer1_size=128, t_max=5):

        self.env = PortfolioEnv()
        self.network = ActorCritic(input_dims=self.env.state_shape(), n_actions=self.env.n_actions(),
                                   gamma=gamma, fc1_dims=layer1_size, lr=alpha)

        if load:
            self.network.load_checkpoint()

        pipes = [Pipe() for i in range(n_agents)]
        self.local_conns, remote_conns = list(zip(*pipes))
        lock = Lock()
        self.workers = [Agent(self.network,
                        lock,
                        conn=remote_conns[i],
                        name=f'w{i}',
                        t_max=t_max) for i in range(n_agents)]

    def train(self):
        [w.start() for w in self.workers]

        self.network.done = False
        while not self.network.done:
            losses = []
            for c in self.local_conns:
                self.network.set_memory(*c.recv())
                losses.append(self.network.calc_loss())
            total_loss = sum(losses)
            self.network.zero_grad()
            total_loss.backward()
            self.network.optimizer.step()
            [c.send('resume') for c in self.local_conns]

        time.sleep(1)
        self.network.save_checkpoint()
