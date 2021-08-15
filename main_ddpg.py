from agents.agent_ddpg import Agent
from env.environment import PortfolioEnv
import numpy as np
from plot import add_curve, save_plot


def main():
    figure_file = 'plots/ddpg.png'

    env = PortfolioEnv(action_scale=1000)
    djia_history = env.get_djia_history()
    add_curve(djia_history/djia_history[0], 'DJIA')

    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=env.state_shape(), tau=0.001, env=env,
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=env.n_actions())

    # agent.load_models()
    np.random.seed(0)

    score_history = []

    observation = env.reset()
    done = False
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info, wealth = env.step(action)
        agent.remember(observation, action, reward, observation_, int(done))
        agent.learn()
        observation = observation_
        print(f"Date: {info},\tBalance: {int(observation[0])},\tWealth: {int(wealth)},\t"
              f"Shares: {observation[31:61]}")
        score_history.append(wealth/1000000)

    agent.save_models()

    add_curve(score_history, 'DDPG')
    save_plot(figure_file)



if __name__ == '__main__':
    main()
