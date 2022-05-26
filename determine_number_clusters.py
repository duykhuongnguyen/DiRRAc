import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt

from utils import pad_ones, train_theta
from mace.loadModel import loadModelForDataset



fig, axs = plt.subplots(1, 3, figsize=(25, 5.5))


SMALL_SIZE = 8
MEDIUM_SIZE = 24
BIGGER_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rcParams.update({'font.size': 36})
plt.tight_layout()


datasets = ['german', 'sba', 'student']
datasets_map = {
        'german': "German",
        'sba': "SBA",
        'student': "Student",
        }
k_list = [i for i in range(1, 21)]


for i in range(len(datasets)):
    dataset_string = datasets[i]
    model_trained, X_train, y_train, X_test, y_test, X_shift, y_shift = loadModelForDataset('lr', dataset_string)
    theta = train_theta(pad_ones(X_train), y_train, 100, return_all=True)
    model = KMeans()
    visualizer = KElbowVisualizer(model, ax=axs[i], k=(2, 10), timings=False)

    visualizer.fit(theta)
    # visualizer.show()
    # sse = {}
    # for k in k_list:
    #     kmeans = KMeans(n_clusters=k, max_iter=1000).fit(theta)
        # data["clusters"] = kmeans.labels_
        #print(data["clusters"])
    #     sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

    axs[i].set_xlabel("Number of components", fontdict={'fontsize': 24})
    axs[i].set_title(datasets_map[datasets[i]])
    
    if i == 0:
        axs[i].set_ylabel("Distortion", fontdict={'fontsize': 30})

plt.tight_layout()

plt.savefig("result/number_cluster.pdf", dpi=1000)
plt.show()
