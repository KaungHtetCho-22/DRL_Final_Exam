import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(scores, losses, figure_file):
    running_avg = np.zeros(len(scores))
    max_idx = 0
    max_reward = -100
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-9):(i+1)])
        if running_avg[i] > max_reward:
            max_reward = running_avg[i]
            max_idx = i
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(running_avg, label='reward')
    plt.plot(max_idx, running_avg[max_idx], 'go', label='max reward')
    plt.title('Moving average reward of last 10 episodes')
    plt.xlabel('Number of Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.plot(losses)
    plt.title('Training loss')
    plt.xlabel('Number of Episode')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(figure_file)

def plot_testing_curve(scores, figure_file):
    mean_scr = np.mean(scores)
    max_scr = max(scores)
    min_scr = min(scores)
    print('mean score: {:.2f}, max score: {:.2f}, min score: {:.2f}'.format(mean_scr, max_scr, min_scr))
    mean_score = [mean_scr for _ in range(len(scores))]
    max_score = [max_scr for _ in range(len(scores))]
    min_score = [min_scr for _ in range(len(scores))]
    plt.figure(figsize=(8,5))
    plt.plot(scores, 'b-', label='reward')
    plt.plot(mean_score, 'y--', label='mean reward')
    plt.plot(max_score, 'g--', label='max reward')
    plt.plot(min_score, 'r--', label='min reward')
    plt.title('Episode reward in Testing mode')
    plt.xlabel('Number of Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(figure_file)

def plot_formation_error(errors, figure_file):
    x = [i+1 for i in range(len(errors[0]))]
    plt.figure(figsize=(8, 5))
    for k in range(len(errors)):
        plt.plot(x, errors[k], label='line %s'%(k+1))
    plt.title('Error of Formation Control')
    plt.xlabel('Number of Step')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(figure_file)

def plot_motion_error(errors, figure_file):
    ep_num = len(errors)
    mean_err = np.mean(errors)
    max_err = max(errors)
    min_err = min(errors)
    print('mean error: {:.2f}, max error: {:.2f}, min error: {:.2f}'.format(mean_err, max_err, min_err))
    ep = [i+1 for i in range(ep_num)]
    mean_error = [mean_err for _ in range(ep_num)]
    max_error = [max_err for _ in range(ep_num)]
    min_error = [min_err for _ in range(ep_num)]
    plt.figure(figsize=(8, 5))
    plt.plot(ep, errors, 'b-', label='error')
    plt.plot(ep, mean_error, 'y--', label='mean error')
    plt.plot(ep, max_error, 'r--', label='max error')
    plt.plot(ep, min_error, 'g--', label='min error')
    plt.title('Motion Control Error Plot')
    plt.xlabel('Number of Episode')
    plt.ylabel('Average Error of Episode')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(figure_file)