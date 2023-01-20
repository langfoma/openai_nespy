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
import tqdm
import warnings

import mario_rl

# get size of terminal
TERMINAL_COLS, TERMINAL_ROWS = os.get_terminal_size()

# get timestamp for now
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

# set defaults
DEFAULT_AGENT_PATH = None
DEFAULT_AGENT_ACTIONS = 'simple-right-only'
DEFAULT_AGENT_ACTIONS_CHOICES = ['complex-movement', 'right-only', 'simple-movement', 'simple-right-only']
DEFAULT_AGENT_TYPE_CHOICES = ['clipped-double-dqn', 'dqn', 'double-dqn']
DEFAULT_AGENT_TYPE = 'double-dqn'
DEFAULT_BATCH_SIZE = 32
DEFAULT_BURN_IN_STEPS = 100000
DEFAULT_DISCOUNT_FACTOR = 0.9
DEFAULT_EXPLORATION_RATE = 1.0
DEFAULT_EXPLORATION_RATE_DECAY = 0.99999975
DEFAULT_EXPLORATION_RATE_MIN = 0.1
DEFAULT_FRAME_SIZE = 84
DEFAULT_FRAME_SKIP = 4
DEFAULT_LEARNING_INTERVAL = 3
DEFAULT_LEARNING_RATE = 0.00025
DEFAULT_MEMORY_LIMIT = 100000
DEFAULT_NUM_EPISODES = 40000
DEFAULT_RANDOM_SEED = 0
DEFAULT_REFRESH_RATE = 24
DEFAULT_RESULTS_DIR = os.path.join('../src/mario_rl', 'results', TIMESTAMP)
DEFAULT_SAVE_INTERVAL = 100

# manage command-line arguments
ARG_PARSER = argparse.ArgumentParser()
ARG_PARSER.add_argument('agent_path', nargs='?', type=str, default=DEFAULT_AGENT_PATH)
ARG_PARSER.add_argument('--agent-actions', type=str.lower, choices=DEFAULT_AGENT_ACTIONS_CHOICES, default=DEFAULT_AGENT_ACTIONS)
ARG_PARSER.add_argument('--agent-type', type=str.lower, choices=DEFAULT_AGENT_TYPE_CHOICES, default=DEFAULT_AGENT_TYPE)
ARG_PARSER.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
ARG_PARSER.add_argument('--burn-in-steps', type=int, default=DEFAULT_BURN_IN_STEPS)
ARG_PARSER.add_argument('--discount-factor', type=float, default=DEFAULT_DISCOUNT_FACTOR)
ARG_PARSER.add_argument('--exploration-rate', type=float, default=DEFAULT_EXPLORATION_RATE)
ARG_PARSER.add_argument('--exploration-rate-decay', type=float, default=DEFAULT_EXPLORATION_RATE_DECAY)
ARG_PARSER.add_argument('--exploration-rate-min', type=float, default=DEFAULT_EXPLORATION_RATE_MIN)
ARG_PARSER.add_argument('--frame-size', type=int, default=DEFAULT_FRAME_SIZE)
ARG_PARSER.add_argument('--frame-skip', type=int, default=DEFAULT_FRAME_SKIP)
ARG_PARSER.add_argument('--learning-interval', type=int, default=DEFAULT_LEARNING_INTERVAL)
ARG_PARSER.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
ARG_PARSER.add_argument('--memory-limit', type=int, default=DEFAULT_MEMORY_LIMIT)
ARG_PARSER.add_argument('--memory-prioritized', action='store_true')
ARG_PARSER.add_argument('--num-episodes', type=int, default=DEFAULT_NUM_EPISODES)
ARG_PARSER.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED)
ARG_PARSER.add_argument('--refresh-rate', type=float, default=DEFAULT_REFRESH_RATE)
ARG_PARSER.add_argument('--replay', action='store_true')
ARG_PARSER.add_argument('--results-dir', type=str, default=DEFAULT_RESULTS_DIR)
ARG_PARSER.add_argument('--save-interval', type=int, default=DEFAULT_SAVE_INTERVAL)
ARG_PARSER.add_argument('--save-memory', action='store_true')


def main(args):
    # set random seed if given
    if args.random_seed is not None:
        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)

    # create results directory
    if not args.replay and args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

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
    env = mario_rl.utils.wrappers.SkipFrame(env, skip=args.frame_skip)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=(args.frame_size, args.frame_size))
    env = gym.wrappers.FrameStack(env, num_stack=args.frame_skip)
    env = gym.wrappers.TransformObservation(env, f=lambda x: x.__array__() / 255.0)
    env = gym.wrappers.TransformObservation(env, f=lambda x: np.moveaxis(x, -1, 1))
    env = gym.wrappers.TransformObservation(env, f=lambda x: np.reshape(x, (-1, args.frame_size, args.frame_size)))
    env.reset()

    # init agent
    if args.agent_path is not None and args.agent_path != '':
        agent = mario_rl.learning.agent.load_agent(args.agent_path)
    else:
        # map agent types to classes
        agent_options = {
            'clipped-double-dqn':   mario_rl.learning.agent.ClippedDoubleDQNAgent,
            'dqn':                  mario_rl.learning.agent.DQNAgent,
            'double-dqn':           mario_rl.learning.agent.DoubleDQNAgent,
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
        agent_path = os.path.join(args.results_dir, 'mario_{}.pt'.format(args.agent_type))
        train(env, agent, agent_path, args)


def train(env, agent, file_path, args):

    # iterate all episodes
    num_episodes = max(0, args.num_episodes)
    pbar = tqdm.tqdm(range(num_episodes), bar_format='{desc} {percentage:3.0f}%|{bar}{r_bar}')
    pbar.set_description_str('{} ({}, {}, {})'.format(
        f'{"[Training]":>12}',
        f'loss: {0:>7.4f}',
        f'q-value: {0:>7.4f}',
        f'reward: {0:>5.0f}',
    ))
    for i in pbar:
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

        # update progress bar
        pbar_loss = agent.logger.value('eps_loss', history=args.save_interval)
        pbar_q_value = agent.logger.value('eps_q_value', history=args.save_interval)
        pbar_reward = agent.logger.value('eps_reward', history=args.save_interval)
        pbar.set_description_str('{} ({}, {}, {})'.format(
            f'{"[Training]":>12}',
            f'loss: {pbar_loss:>7.4f}',
            f'q-value: {pbar_q_value:>7.4f}',
            f'reward: {pbar_reward:>5.0f}',
        ))

        # save to filesystem
        if i % args.save_interval == 0:
            agent.save(file_path, save_log=True, save_memory=args.save_memory)

    # save agent
    agent.save(file_path, save_log=True, save_memory=args.save_memory)


def replay(env, agent, args):
    # print header
    text = 'Replaying'
    print('{}[{}]{}'.format('-' * 2, text, '-' * max(0, TERMINAL_COLS - len(text) - 4)))

    # determine max length of episode based on history
    eps_length_max = agent.logger.value('eps_length', history=args.save_interval)

    # iterate all episodes
    num_episodes = max(0, args.num_episodes)
    for i in range(num_episodes):
        # reset environment
        state = env.reset()

        # init episode metrics
        eps_length = 0
        eps_reward = 0

        # iterate through current episode
        pbar = tqdm.tqdm(total=eps_length_max, bar_format='{desc} {percentage:3.0f}%|{bar}{r_bar}')
        pbar.set_description_str('{} ({}, {})'.format(
            f'{f"[Playing {i}/{num_episodes}]":>12}',
            f'length: {0:>5.0f}',
            f'reward: {0:>5.0f}',
        ))
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
            pbar.update(1)
            pbar.set_description_str('{} ({}, {})'.format(
                f'{"[Playing]":>12}',
                f'length: {eps_length:>5.0f}',
                f'reward: {eps_reward:>5.0f}',
            ))


if __name__ == '__main__':
    # suppress warnings
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # suppress numpy warnings
    np.seterr(over='ignore')

    # run main function
    main(ARG_PARSER.parse_args())
