import numpy as np
import matplotlib.pyplot as plt
from .cluster import define_distance, correlation_distances

def plot_robustness(robustness, ax):
    D_l = []
    sim_l = []
    k_l = []
    for robust_iter in robustness:
        D = robust_iter.shape[1]
        for k in range(D):
            for sim in robust_iter[:,k]:
                D_l.append(D)
                sim_l.append(sim)
                k_l.append(k)
                
    import pandas as pd
    import seaborn as sns
    arr = np.array([sim_l,D_l,k_l])
    df = pd.DataFrame(arr.T)
    df.columns = ['sim', 'D', 'k']
    df['sim'] = df['sim'].astype(float)    

    plotting_parameters = {
        'data':    df,
        'y':       'D',
        'x':       'sim',
        'hue':       'k',
        'orient':       'h',
        'palette': "colorblind"
    }

    return sns.barplot(ax=ax, **plotting_parameters)

def plot_percent_pass(percent_pass, ax):
    D_l = []
    sim_l = []
    k_l = []
    for percent_pass_i in percent_pass:
        D = len(percent_pass_i)
        for k in range(D):
            sim_l.append(percent_pass_i[k])
            k_l.append(k)
            D_l.append(D)
                
    import pandas as pd
    import seaborn as sns
    arr = np.array([sim_l,D_l,k_l])
    df = pd.DataFrame(arr.T)
    df.columns = ['sim', 'D', 'k']
    df['sim'] = df['sim'].astype(float)    

    plotting_parameters = {
        'data':    df,
        'y':       'D',
        'x':       'sim',
        'hue':       'k',
        'orient':       'h',
        'palette': "colorblind"
    }

    return sns.barplot(ax=ax, **plotting_parameters)


def silhouette(X, caps, plot=False):
    ############################### SILHOUETTE PLOT ###############################################
    # code from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    import matplotlib.cm as cm
    from sklearn.metrics import silhouette_samples
    # Compute the silhouette scores for each sample; silhouette = (b-a)/max(a,b), where 'a' is 
    # the average distance of the data point relative to other point within the same cluster, 
    # and 'b' is the distance relative to the nearest neighbor cluster
    
    labels = caps.labels_
    sample_distances = define_distance(caps.distance)(X,X)
    sample_distances[np.where(sample_distances<0)] = 0 # clip negative values, which can happen with correlation because numerical instability
    np.fill_diagonal(sample_distances,0)

    # Compute the silhouette scores for each sample
    sample_silhouette = silhouette_samples(sample_distances, labels, metric='precomputed')
    silhouette_avg = sample_silhouette.mean()
    
    if plot:
        k = labels.max()+1
        # Create silhouette plot
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(6, 4)
        ax1.set_xlim([-0.5, 1])
        ax1.set_ylim([0, len(X) + (k + 1) * 10]) # plots of individual clusters, to demarcate them clearly.
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette[labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(f"Silhouette plot \n (mean = {str(np.around(silhouette_avg,3))})", fontsize=15)
        ax1.set_xlabel("Sample silhouette", fontsize=12)
        ax1.set_ylabel("Cluster label", fontsize=12)

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    return sample_silhouette, silhouette_avg


def cluster2data_corr(X, caps, corr_threshold = 0.5):
    distances = correlation_distances(X=X, Y=caps.cluster_centers_) # using correlation so its interpretable across methods
    cluster_corr = (distances - 1)*-1 # convert back to correlation
    labels = np.argmin(distances, axis=1)
    within_cluster_corr = [cluster_corr[:,i][labels==i] for i in range(labels.max()+1)] # compute centroid correlation from each point within a cluster
    percent_pass = [(within_cluster_corr_i>corr_threshold).sum()/len(within_cluster_corr_i) for within_cluster_corr_i in within_cluster_corr]
    return within_cluster_corr, percent_pass

def plot_cluster2data_corr(X, caps, corr_threshold = 0.5):
    import matplotlib.cm as cm
    
    within_cluster_corr, percent_pass = cluster2data_corr(X, caps, corr_threshold)

    k = len(percent_pass)
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(6, 4)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, len(X) + (k + 1) * 10]) # plots of individual clusters, to demarcate them clearly.
    y_lower = 10
    for i in range(k):
        ith_cluster_corr_values = within_cluster_corr[i]

        ith_cluster_corr_values.sort()

        size_cluster_i = ith_cluster_corr_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_corr_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title(f"Sample-to-cluster correlation \n (mean percent threshold pass = {str(np.around(np.array(percent_pass).mean(),3))})", fontsize=15)
    ax1.set_xlabel("Correlation", fontsize=12)
    ax1.set_ylabel("Cluster label", fontsize=12)

    ax1.axvline(x=corr_threshold, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks


def stability_report(caps_l, X, k_range, pass_threshold=0.5):

    for caps in caps_l:
        if caps.labels_ is None or len(caps.labels_)!=X.shape[0]:
            print("\r","Changing labels to match input X.", end="")
            caps.labels_ = caps.predict(X)

    VE = [caps.estimate_R2(X) for caps in caps_l]
    robustness = [caps.cross_rep_sim for caps in caps_l]
    from sklearn.metrics import calinski_harabasz_score
    var_ratio = [calinski_harabasz_score(X, caps.labels_) for caps in caps_l]
    silhouette_avg = [silhouette(X, caps, plot=False)[1] for caps in caps_l]
    percent_pass = [cluster2data_corr(X, caps, pass_threshold)[1] for caps in caps_l]
    percent_pass_mean = np.array([np.array(percent_pass_i).mean() for percent_pass_i in percent_pass])
    percent_pass_std = np.array([np.array(percent_pass_i).std() for percent_pass_i in percent_pass])

    
    fig1,axes = plt.subplots(1,4, figsize=(12,3), constrained_layout = True)
    axes[0].plot(k_range, VE)
    axes[0].set_xlabel('Number of clusters', fontsize=12)
    axes[0].set_ylabel('Variance explained (R2)', fontsize=12)
    axes[0].set_ylim([0,1])

    axes[1].plot(k_range, var_ratio)
    axes[1].set_xlabel('Number of clusters', fontsize=12)
    axes[1].set_ylabel('Cluster variance ratio', fontsize=12)

    axes[2].plot(k_range, silhouette_avg)
    axes[2].set_xlabel('Number of clusters', fontsize=12)
    axes[2].set_ylabel('Silhouette average', fontsize=12)

    axes[3].plot(k_range, percent_pass_mean)
    axes[3].fill_between(k_range, percent_pass_mean - percent_pass_std, percent_pass_mean + percent_pass_std, alpha=0.2)
    axes[3].set_xlabel('Number of clusters', fontsize=12)
    axes[3].set_ylabel('Percent data2cluster', fontsize=12)
    axes[3].set_ylim([0,1])
    
    l = len(k_range)
    fig2,ax = plt.subplots(1,1, figsize=(4,l*2), constrained_layout = True)
    ax = plot_robustness(robustness, ax)
    ax.set_xlabel('Number of clusters', fontsize=12)
    ax.set_ylabel('Cross-repetition robustness', fontsize=12)
    vmax = np.array([z for i in robustness for j in i for z in j]).max()
    ax.set_xlim([0,vmax])
    
    l = len(k_range)
    fig3,ax = plt.subplots(1,1, figsize=(4,l*2), constrained_layout = True)
    ax = plot_percent_pass(percent_pass, ax)
    ax.set_title(f"Sample2Cluster percentage similar", fontsize=15)
    ax.set_xlabel('Number of clusters', fontsize=12)
    ax.set_ylabel(f'Percent above {pass_threshold} correlation', fontsize=12)
    vmax = np.array([z for i in robustness for j in i for z in j]).max()
    ax.set_xlim([0,vmax])
    
    return fig1,fig2,fig3
