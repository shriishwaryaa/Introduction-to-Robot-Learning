import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if len(X) > 120:
            break
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    # logdir = os.path.join(args.logdir, 'events*')
    # eventfile = glob.glob(logdir)[0]

    # X, Y = get_section_results(eventfile)
    # for i, (x, y) in enumerate(zip(X, Y)):
    #     print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

    logdir_dqn = os.path.join(args.logdir, 'dqn*')
    logdir_ddqn = os.path.join(args.logdir, 'ddqn*')

    Y_dqn = []
    for eventfile in glob.glob(logdir_dqn):
        X, Y = get_section_results(eventfile)
        Y_dqn.append(Y)

    Y_ddqn = []
    for eventfile in glob.glob(logdir_ddqn):
        X, Y = get_section_results(eventfile)
        Y_ddqn.append(Y)

    Y_dqn = np.array(Y_dqn)
    Y_ddqn = np.array(Y_ddqn)

    Y_dqn_mean = np.mean(Y_dqn, axis=0)
    Y_dqn_std = np.std(Y_dqn, axis=0)

    Y_ddqn_mean = np.mean(Y_ddqn, axis=0)
    Y_ddqn_std = np.std(Y_ddqn, axis=0)

    X = X[:-1]

    plt.plot(X, Y_dqn_mean, label='DQN')
    plt.fill_between(X, Y_dqn_mean - Y_dqn_std, Y_dqn_mean + Y_dqn_std, alpha=0.2)
    plt.plot(X, Y_ddqn_mean, label='DDQN')
    plt.fill_between(X, Y_ddqn_mean - Y_ddqn_std, Y_ddqn_mean + Y_ddqn_std, alpha=0.2)
    plt.xlabel('Timesteps')
    plt.ylabel('Average return')
    plt.title('DQN vs Double DQN on LunarLander\n (shaded regions are standard deviation)')
    plt.legend()
    plt.savefig('dqn_vs_ddqn.png')
    plt.show()

