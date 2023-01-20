import argparse
import datetime
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import os
import random
import time
import torch

from mario_rl.learning.agent import ClippedDoubleDQNAgent, DQNAgent, DoubleDQNAgent, load_agent
from mario_rl.utils.progress import ProgressDisplay
from mario_rl.utils.wrappers import SkipFrame

# get size of terminal
TERMINAL_COLS, TERMINAL_ROWS = os.get_terminal_size()


def main(args):
    # set random seed if given
    if args.random_seed is not None:
        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)

    # create results directory
    if not args.replay and not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # set available actions
    action_options = {
        'complex-movement':     COMPLEX_MOVEMENT,
        'right-only':           RIGHT_ONLY,
        'simple-movement':      SIMPLE_MOVEMENT,
        'simple-right-only':    [['right'], ['right', 'A']],
    }
    if args.agent_actions not in action_options:
        raise ValueError('Invalid actions.')
    actions = action_options[args.agent_actions]

    # init environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, actions)
    env = SkipFrame(env, skip=args.frame_skip)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=(args.frame_size, args.frame_size))
    env = gym.wrappers.FrameStack(env, num_stack=args.frame_skip)
    env = gym.wrappers.TransformObservation(env, f=lambda x: x.__array__() / 255.0)
    env = gym.wrappers.TransformObservation(env, f=lambda x: np.moveaxis(x, -1, 1))
    env = gym.wrappers.TransformObservation(env, f=lambda x: np.reshape(x, (-1, args.frame_size, args.frame_size)))
    env.reset()

    # init agent
    if args.agent_path is not None and args.agent_path != '':
        agent = load_agent(args.agent_path)
    else:
        # map agent types to classes
        agent_options = {
            'clipped-double-dqn':   ClippedDoubleDQNAgent,
            'dqn':                  DQNAgent,
            'double-dqn':           DoubleDQNAgent,
        }
        if args.agent_type not in agent_options:
            raise ValueError('Invalid agent type.')

        # create instance of agent
        agent = agent_options[args.agent_type](
                state_shape=(args.frame_skip, args.frame_size, args.frame_size),
                actions=actions,
                batch_size=args.batch_size,
                burn_in_steps=args.burn_in_steps,
                discount_factor=args.discount_factor,
                exploration_rate=args.exploration_rate,
                exploration_rate_decay=args.exploration_rate_decay,
                exploration_rate_min=args.exploration_rate_min,
                learning_interval=args.learning_interval,
                learning_rate=args.learning_rate,
                memory_limit=args.memory_limit,
                memory_prioritized=args.memory_prioritized,
            )

    # print description of agent
    print(agent.description(width=TERMINAL_COLS))

    if args.replay:
        # replay agent
        replay(env, agent, args)
    else:
        # train agent
        agent_path = os.path.join(args.results_dir, 'mario_{}.pth'.format(args.agent_type))
        train(env, agent, agent_path, args)


def train(env, agent, file_path, args):
    # print header
    text = 'Training'
    print('{}[{}]{}'.format('-' * 2, text, '-' * max(0, TERMINAL_COLS - len(text) - 4)))

    # init progress bar
    progress = ProgressDisplay(bar_length=0)

    # iterate all episodes
    n_episodes = max(0, args.n_episodes)
    for i in range(n_episodes):
        # reset environment
        state = env.reset()

        # init episode metrics
        eps_length = 0
        eps_learn_count = 0
        eps_loss = 0
        eps_q_value = 0
        eps_reward = 0

        # iterate through current episode
        done = False
        while not done:
            # make agent interact with environment
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            done = done or info['flag_get']

            # make agent learn from experience
            q_value, loss = agent.learn(state, action, next_state, reward, done)
            state = next_state

            # update episode metrics
            eps_length += 1
            eps_reward += reward
            if q_value is not None:
                eps_q_value += q_value
                eps_loss += loss
                eps_learn_count += 1

        # compute metric means
        eps_q_value = eps_q_value / max(1, eps_learn_count)
        eps_loss = eps_loss / max(1, eps_learn_count)

        # log metrics for current episode
        agent.logger.log(
            steps=agent.steps,
            exploration_rate=agent.exploration_rate,
            eps_length=eps_length,
            eps_loss=eps_loss,
            eps_q_value=eps_q_value,
            eps_reward=eps_reward,
        )

        # update progress
        progress.update(i + 1, n_episodes, suffix_text=' - '.join([
            'loss: {:.4f}'.format(agent.logger.value('eps_loss', history=args.save_interval)),
            'q-value: {:.4f}'.format(agent.logger.value('eps_q_value', history=args.save_interval)),
            'reward: {:.0f}'.format(agent.logger.value('eps_reward', history=args.save_interval)),
        ]))

        # save to filesystem
        if i % args.save_interval == 0:
            agent.save(file_path, save_log=True, save_memory=args.save_memory)

    # terminate progress bar and save agent
    progress.terminate()
    agent.save(file_path, save_log=True, save_memory=args.save_memory)


def replay(env, agent, args):
    # print header
    text = 'Replaying'
    print('{}[{}]{}'.format('-' * 2, text, '-' * max(0, TERMINAL_COLS - len(text) - 4)))

    # determine max length of episode based on history
    eps_length_max = agent.logger.value('eps_length', history=args.save_interval)

    # init progress bar
    progress = ProgressDisplay(bar_length=20, show_steps=False, show_time=False)

    # iterate all episodes
    n_episodes = max(0, args.n_episodes)
    for i in range(n_episodes):
        # reset environment
        state = env.reset()

        # init episode metrics
        eps_length = 0
        eps_reward = 0

        # iterate through current episode
        last_update_time = time.time()
        done = False
        while not done:
            # update at refresh rate
            elapsed_time = time.time() - last_update_time
            if elapsed_time < (1.0 / max(1.0, args.refresh_rate)):
                continue

            # perform action on environment
            action = agent.act(state, exploration=False)
            state, reward, done, info = env.step(action)
            done = done or info['flag_get']

            # render environment
            env.render()

            # update episode metrics
            eps_length += 1
            eps_reward += reward

            # update time for refresh
            last_update_time = time.time()

            # update progress
            progress.update(eps_length, eps_length_max, suffix_text=' - '.join([
                'length: {:.0f}'.format(eps_length),
                'reward: {:.0f}'.format(eps_reward),
            ]))

    # terminate progress bar
    progress.terminate()


if __name__ == '__main__':
    # suppress warnings
    import warnings
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress numpy warnings
    np.seterr(over='ignore')

    # create timestamp for now
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    # manage command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_path', nargs='?', type=str, default=None)
    parser.add_argument('--agent-actions', type=str.lower,
                        choices=['complex-movement', 'right-only', 'simple-movement', 'simple-right-only'],
                        default='simple-right-only')
    parser.add_argument('--agent-type', type=str.lower,
                        choices=['clipped-double-dqn', 'dqn', 'double-dqn'],
                        default='double-dqn')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--burn-in-steps', type=int, default=100000)
    parser.add_argument('--discount-factor', type=float, default=0.9)
    parser.add_argument('--exploration-rate', type=float, default=1.0)
    parser.add_argument('--exploration-rate-decay', type=float, default=0.99999975)
    parser.add_argument('--exploration-rate-min', type=float, default=0.1)
    parser.add_argument('--frame-size', type=int, default=84)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--learning-interval', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.00025)
    parser.add_argument('--memory-limit', type=int, default=100000)
    parser.add_argument('--memory-prioritized', action='store_true')
    parser.add_argument('--n-episodes', type=int, default=40000)
    parser.add_argument('--random-seed', type=int, default=None)
    parser.add_argument('--refresh-rate', type=float, default=24)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--results-dir', type=str, default=os.path.join('../src/mario_rl', 'results', timestamp))
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--save-memory', action='store_true')

    # run main function
    main(parser.parse_args())
