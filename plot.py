import matplotlib.pyplot as plt
import glob
from collections import deque
import numpy as np
from utils import random_color
import argparse

GOAL_SCORE=30.0
LINESTYLES = ['-', '--', ':', '-.']

parser=argparse.ArgumentParser(description="Plot the statistics of training traces")
parser.add_argument('-l', '--list_of_models', type=str, metavar='', required=True, help="specify model(s) to plot. For multiple model, specify by a string with comma, for example, model1,model2,.. One can specify at most four models")
args=parser.parse_args()


def load_training_trace_as_list(file_path):
    """
    to load training trace of a given agent,
    01.02.2019: currently only support for saved data--string format of lists
    :param file_path: the path to the file
    :return:
    """
    with open(file_path, 'r') as file:
        return eval(file.readline())


def min_len_list_of_lists(list_of_lists):
    """
    find minimal length of a given list of lists
    :param list_of_lists: a list of lists with potentially variable length
    :return: the minimal list-length in the list of lists
    """
    return min(map(len, list_of_lists))


def plot_avg_scores_with_confidence(scores, model_name, color=random_color(), linestyle='-'):
    l = min_len_list_of_lists(scores)
    scores = [score[:l] for score in scores]

    mean_score = np.mean(scores, axis=0)
    std_score = np.std(scores, axis=0)

    x = np.arange(l)
    n_exps = len(scores)
    plt.plot(x, mean_score, color=color, label=model_name, linestyle=linestyle)
    plt.fill_between(x, mean_score - 2*std_score / np.sqrt(n_exps),
                        mean_score + 2*std_score / np.sqrt(n_exps),
                        alpha=0.2, color=color)


def window_avg_scores(scores, length=100):
    # load a single trace of scores and convert them into windowed average scores
    scores_window = deque(maxlen=length)
    windowed_avg=[]
    for score in scores:
        scores_window.append(score)
        windowed_avg.append(np.mean(scores_window))
    return windowed_avg


def get_plotting_data(list_of_models):
    """
    process the progress.txt files in data directory to get out model performance scores
    in dict format:
        {"model": {"scores": ...,
                   "window-avg-scores" ...}}
    :param list_of_models: a list of models' name to report
    :return: a dict collecting all
    """
    scores_dict = {}
    for i, model in enumerate(list_of_models):
        data_paths = glob.glob('./data/{}*/progress.txt'.format(model))
        data_paths = sorted(data_paths)

        scores = []
        avg_scores = []

        for data_path in data_paths:
            score = load_training_trace_as_list(data_path)
            scores.append(score)
            avg_scores.append(window_avg_scores(score))

        scores_dict[model] = {"scores": scores,
                              "window-avg-scores": avg_scores}
        
    return scores_dict


def plot_avg_training_traces(scores_dict):
    fig = plt.figure(figsize=(20, 30))
    for i, model in enumerate(list(scores_dict.keys())):
        tmp=scores_dict[model]
        if i<=3:
            # currently, only support for different linestyles for at most four models
            linestyle=LINESTYLES[i]
        else:
            linestyle='-'
        plt.subplot(211)
        plot_avg_scores_with_confidence(tmp['scores'], model_name=model,
                                        color=random_color(i), linestyle=linestyle)
        plt.subplot(212)
        plot_avg_scores_with_confidence(tmp['window-avg-scores'], model_name=model,
                                        color=random_color(i), linestyle=linestyle)

    plt.xlabel('#Episodes')
    plt.subplot(211)
    plt.ylabel('score')
    plt.axhline(y=GOAL_SCORE, color='y', linestyle='--', label="goal_score")
    plt.subplot(212)
    plt.ylabel('avg-100-score')
    plt.axhline(y=GOAL_SCORE, color='y', linestyle='--', label="goal_score")
    plt.ylim(top=GOAL_SCORE + 1.0)
    plt.legend()

    return fig


def plot_best_training_traces(scores_dict):
    fig = plt.figure(figsize=(20,30))
    for i, model in enumerate(list(scores_dict.keys())):
        j = np.argmin([len(scores) for scores in scores_dict[model]['scores']]) # find the most efficient (least time spent) trial
        scores=scores_dict[model]['scores'][j]
        avg_scores=scores_dict[model]['window-avg-scores'][j]
        x=np.arange(len(scores))
        plt.subplot(211)
        plt.plot(x, scores, label=model, color=random_color(i))
        plt.subplot(212)
        plt.plot(x, avg_scores, label=model, color=random_color(i))

    plt.xlabel('#Episodes')
    plt.subplot(211)
    plt.ylabel('score')
    plt.axhline(y=GOAL_SCORE, color='y', linestyle='--', label="goal_score")
    plt.subplot(212)
    plt.ylabel('avg-100-score')
    plt.axhline(y=GOAL_SCORE, color='y', linestyle='--', label="goal_score")
    plt.ylim(top=GOAL_SCORE + 1.0)
    plt.legend()
    return fig


if __name__ == "__main__":
    # 1. formatting data as a dict
    print(args.list_of_models.split(','))
    all_scores = get_plotting_data(args.list_of_models.split(','))
    # 2. get plots
    # 2.1 plot avg
    plot_avg_training_traces(all_scores)
    # 2.2 plot the best performing agent
    plot_best_training_traces(all_scores)
    plt.show()