#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy
from pyksc import ksc
from pyksc import metrics
from csv import DictReader, DictWriter
import matplotlib.pyplot as plt

def load_discussions(csv_filepath):
    csv_file = open(csv_filepath , 'r')
    discussions = DictReader(csv_file)

    time_series = []
    projects = []

    for discussion in discussions:
    	discussion_time_series = [discussion[column] for column in discussions.fieldnames if column not in ['project']]
        time_series.append(discussion_time_series)
        projects.append(discussion['project'])

    time_series = numpy.array(time_series, dtype=numpy.double)
    
    return projects, time_series

def get_clusters(time_series, k):
    clusters = {}
    centroids, assign, best_shift, cent_dists = ksc.ksc(time_series, k)

    if assign is not None:
        for series, cluster in zip(time_series, assign):
            if cluster in clusters.keys():
                clusters[cluster].append(series)
            else:
                clusters[cluster] = [series]

    return clusters, centroids, assign, best_shift, cent_dists

def get_beta_cv(time_series, min_cl = 2, max_cl = 16):
    print("Saving βCV values in 'beta_cv.txt' file")

    beta_cv = []
    k_possible_values = [k for k in range(min_cl, max_cl)]

    with open('./beta_cv.txt', 'w') as text_file:
        text_file.write('# βCV (for 2 ≤ k ≤ 15):\n')

        for k in k_possible_values:
            _, _, assign, _, _ = get_clusters(time_series, k)
            ratio = metrics.beta_cv(time_series, assign)
            text_file.write(str(k) + ': ' + str(ratio) + '\n')
            beta_cv.append(ratio)

    return beta_cv

def plot_beta_cv(beta_cv):
    k_values = [k for k in range(len(beta_cv))]

    figure = plt.figure()
    plt.plot(k_values, beta_cv, color='black')
    plt.xlabel('# Clusters', fontsize = 14)
    plt.ylabel(r'$\beta$cv', fontsize = 14)
    plt.title(r'$\beta$cv for 2 $\leq$ k $\leq$ 15')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    figure.savefig('./images/beta_cv.png', bbox_inches='tight', format='png', dpi=1000)

def plot_clusters(clusters):
    time_range = None 

    for cluster in clusters.keys():
        figure = plt.figure()
        plt.ylim([0, 1000])
        plt.xlabel('Weeks', fontsize = 16)
        plt.ylabel('# Open Discussions', fontsize = 16)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)

        for time_series in clusters[cluster]:

            # Defines the values for the x axis
            if time_range is None:
                time_range = [-i for i in range(len(time_series) - 1, -1, -1)]
                plt.xlim([min(time_range), max(time_range)])

            plt.plot(time_range, time_series, color=clusters_colors[cluster])

        filename = os.path.join('images/cluster_' + str(cluster) + '.png')

        if os.path.isfile(filename):
            os.remove(filename)

        figure.savefig(filename, bbox_inches='tight', format='png', dpi=1000)

def plot_centroids(centroids):
    time_range = None

    for cluster, centroid in enumerate(centroids):

        if time_range is None:
            time_range = [-i for i in range(len(centroid) - 1, -1, -1)]

        figure = plt.figure()
        plt.ylim([0, 1.0])
        plt.xlim([min(time_range), max(time_range)])
        plt.xlabel('Weeks', fontsize=16)
        plt.ylabel('Average', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.plot(time_range, centroid, color=clusters_colors[cluster])
        
        figure.savefig('images/centroid_' + str(cluster) + '.png', bbox_inches='tight', format='png', dpi=1000)

def export_centroids(centroids):
    with open('./centroids.txt', 'w') as text_file:
        print('Saving K-Spectral Centroids (KSC) in "centroids.txt" file')
        text_file.write('\n# K-Spectral Centroids (KSC):\n')

        for cluster, centroid in enumerate(centroids):
            growth_rate = centroid[0] + centroid[-1] * 100
            text_file.write("Centroid for Cluster #" + str(cluster) + "\n")
            text_file.write("Growth Rate: " + str(cluster) + "\n")

            for value in centroid:
                text_file.write(str(value) + '\n')

def export_projects_per_clusters(projects, assign):
    clusters = {}

    for project, cluster in zip(projects, assign):
        if cluster in clusters.keys():
            clusters[cluster].append(project)
        else:
            clusters[cluster] = [project]
    
    for cluster in clusters:
        with open ('projects_in_cluster_' + str(cluster) + '.csv', 'w') as csv_file:
            for project in clusters[cluster]:
                csv_file.write(project + '\n')

if __name__ == '__main__':
    if not os.path.exists('./images'):
        os.makedirs('./images')

    projects, time_series = load_discussions('./discussions.csv')

    # Run this once to get the optimal k for your data
    beta_cv_values = get_beta_cv(time_series)
    plot_beta_cv(beta_cv_values)

    # Use the optimal number for k to get the final clusters
    clusters, centroids, assign, _, _  = get_clusters(time_series, 4)
    # The list below should be of the same size of k
    # You can find other named matplotlib colors at:
    # matplotlib.org/3.5.0/gallery/color/named_colors.html
    export_projects_per_clusters(projects, assign)
    export_centroids(centroids)

    clusters_colors = ['darkorange', 'forestgreen', 'royalblue', 'firebrick']
    plot_clusters(clusters)
    plot_centroids(centroids)
