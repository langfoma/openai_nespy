from nes_py.app.play_human import play_human
import argparse
import datetime
import gym_super_mario_bros
import os
import pickle
import time

from mario_rl.utils.progress import ProgressDisplay


def main(args):
    # init environment (keyboard: WASD OP ENTER/SPACE)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env.reset()

    if not args.replay:
        play(env, args)
    else:
        # load existing recording
        if args.recording_path is not None and args.recording_path != '' and os.path.exists(args.recording_path):
            with open(args.recording_path, 'rb') as f:
                recording = pickle.load(f)

            # replay past recording
            replay(env, recording, args)


def play(env, args):
    # init recording
    recording = dict()
    if args.actions_only:
        recording['action'] = list()
    else:
        recording['state'] = list()
        recording['action'] = list()
        recording['next_state'] = list()
        recording['reward'] = list()
        recording['done'] = list()

    # define recording callback
    def recording_cb(state, action, reward, done, next_state):
        nonlocal recording
        if 'state' in recording:
            recording['state'].append(state)
        if 'action' in recording:
            recording['action'].append(action)
        if 'next_state' in recording:
            recording['next_state'].append(next_state)
        if 'reward' in recording:
            recording['reward'].append(reward)
        if 'done' in recording:
            recording['done'].append(done)
        if done:
            raise StopIteration

    try:
        # render environment for human playback
        play_human(env, callback=recording_cb)
    except KeyboardInterrupt:
        pass
    except StopIteration:
        pass
    finally:
        # save recording to filesystem
        if args.recording_path is not None and args.recording_path != '':
            with open(args.recording_path, 'wb') as f:
                pickle.dump(recording, f)


def replay(env, recording, args):
    # get actions from recording
    if 'action' not in recording:
        return
    actions = recording['action']

    # init progress bar
    progress = ProgressDisplay(bar_length=20, show_steps=False, show_time=False)

    # init episode metrics
    eps_length = 0
    eps_reward = 0

    last_update_time = time.time()
    done = False
    while not done and eps_length < len(actions):
        # enforce refresh delay
        elapsed_time = time.time() - last_update_time
        if args.refresh_rate > 0 and elapsed_time < (1.0 / args.refresh_rate):
            continue

        # perform recorded action on environment
        action = actions[max(0, min(eps_length, len(actions) - 1))]
        next_state, reward, done, info = env.step(action)
        done = done or info['flag_get']

        # render environment
        env.render()

        # shift to next step
        eps_length += 1
        eps_reward += reward

        # update progress
        progress.update(eps_length, len(actions), suffix_text=' - '.join([
            'length: {:.0f}'.format(eps_length),
            'reward: {:.0f}'.format(eps_reward),
        ]))

        # update time for next update
        last_update_time = time.time()

    # terminate progress bar
    progress.terminate()


if __name__ == '__main__':
    # suppress warnings
    import warnings
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    # create timestamp for now
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

    # manage command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('recording_path', nargs='?', type=str, default=None)
    parser.add_argument('--actions-only', action='store_true')
    parser.add_argument('--refresh-rate', type=float, default=24 * 4)
    parser.add_argument('--replay', action='store_true')

    # run main function
    main(parser.parse_args())
