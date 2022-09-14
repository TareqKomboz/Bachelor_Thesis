import os

import numpy as np
import matplotlib.pyplot as plt

from db.performance import read_from_sql_for_correlation_matrix
from definitons import ROOT_DIR
from objective_functions.tf_objective_functions import FUNCTIONS

PLOT_DIR = os.path.join(ROOT_DIR, "comparison_plots")
PLOT_NAME = "rel_corr_mat"

properties = ["Single optimum",
              "Multiple optima",
              "Polynomial",
              "Many local minima",
              "Minima in parabolic shape",
              "Steep hills/walls",
              "Steep valleys",
              "Canyons",
              "(Mostly) flat areas"]

property_relations = np.array(
    ((1, 0, 0, 1, 1, 0, 1, 1),
     (0, 1, 1, 0, 0, 1, 0, 0),
     (0, 1, 0, 0, 1, 1, 1, 0),
     (1, 0, 1, 1, 0, 1, 1, 1),
     (0, 0, 1, 1, 0, 0, 0, 0),
     (0, 0, 1, 1, 0, 0, 0, 1),
     (1, 1, 0, 1, 0, 0, 1, 1),
     (0, 0, 0, 0, 0, 0, 1, 1),
     (1, 0, 0, 0, 0, 0, 0, 1)))


def main():
    df = get_data()
    rel_matrix = build_corr_matrix(df)
    generalization_scores = np.mean(rel_matrix, axis=0)
    solving_scores = np.mean(rel_matrix, axis=1)
    print("relative performance correlation matrix")
    print(np.round(rel_matrix, 3))
    print("relative generalization scores")
    print(np.round(generalization_scores, 3))
    print("ease of solving score")
    print(np.round(solving_scores, 3))
    property_scores = build_property_scores(rel_matrix)
    print("property scores")
    print(np.round(property_scores, 3))
    line = "property,score\n"
    for property, score in zip(properties, property_scores):
        line += "{},{}\n".format(property, np.round(score, 3))
    with open("property_score.csv", 'w') as writer:
        writer.write(line)

def build_property_scores(matrix):
    scores = []
    for relation in property_relations:
        score = 0.0
        indices = np.where(relation == 1)[0]
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                score += matrix[i, j]
        avg_score = score / (len(indices)**2 - len(indices))
        scores.append(avg_score)
    return scores

def build_corr_matrix(df):
    n = len(FUNCTIONS)
    max_perf = [df[df['trained_on'] == function_name][function_name].max() for function_name in FUNCTIONS.keys()]
    abs_matrix = np.zeros((n, n))
    for i, trained_on in enumerate(FUNCTIONS.keys()):
        for j, run_on in enumerate(FUNCTIONS.keys()):
            abs_matrix[i, j] = df[df["trained_on"] == trained_on][run_on].max()
    rel_matrix = np.array([(abs_vector / perf) for (abs_vector, perf) in zip(abs_matrix.T, max_perf)])

    fig, ax = plt.subplots()
    ax.matshow(rel_matrix)

    for (i, j), z in np.ndenumerate(rel_matrix):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center')

    labels = [name[:3] for name in FUNCTIONS.keys()]
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    plt.xlabel("trained on")
    plt.ylabel("run on")
    plt.savefig(os.path.join(PLOT_DIR, PLOT_NAME), dpi=400, transparent=True)
    plt.show()
    return rel_matrix

def get_data():
    return read_from_sql_for_correlation_matrix(0.2)


if __name__ == "__main__":
    main()
