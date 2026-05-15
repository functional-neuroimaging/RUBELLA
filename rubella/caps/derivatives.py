from itertools import groupby
import numpy as np

def convert_to_dwell_time(labels):
    """
    Replace each run of identical labels with its length.
    
    Example: [1, 1, 2, 2, 2, 3] → [2, 2, 3, 3, 3, 1]
    """
    result = []
    for _, group in groupby(labels):
        run = list(group)
        result.extend([len(run)] * len(run))
    return result


def estimate_transitions(labels, censoring_mask=None, transition_step_size=1, exclude_steadystate=False):
    if censoring_mask:
        # because censored frames are set to -1, they are not counted in the transition counts below
        # censored frames break the transition estimation, until there is again two consecutive 
        # uncensored transitions
        filled_labels = np.array([-1]*len(censoring_mask))
        filled_labels[censoring_mask] = labels
    else:
        filled_labels = labels
        
    if exclude_steadystate:
        unsteady_filled_labels = [filled_labels[0]]
        i=0
        while(i+1<len(filled_labels)):
            if filled_labels[i]!=filled_labels[i+1]:
                unsteady_filled_labels.append(filled_labels[i+1])
            i+=1

        filled_labels = np.array(unsteady_filled_labels)
    
    transitions_str = np.array([
        f"{x}_{y}" for x, y in zip(filled_labels[:-transition_step_size], filled_labels[transition_step_size:])
    ])

    k = labels.max()+1
    transition_counts=np.zeros([k,k])
    for x in range(k):
        for y in range(k):
            # rows are the origin CAP, and column the subsequent CAP
            transition_counts[x,y]=(transitions_str==f"{x}_{y}").sum()
    # divide rows by their sum to get transition probability from a CAP to the target
    # i.e. given a certain CAP state, what is the probability of the next CAP
    transition_probability = (transition_counts.T/transition_counts.sum(axis=1)).T
    return transition_counts, transition_probability


def estimate_transitions_3steps(labels, censoring_mask=None, exclude_steadystate=False):
    if censoring_mask:
        # because censored frames are set to -1, they are not counted in the transition counts below
        # censored frames break the transition estimation, until there is again two consecutive 
        # uncensored transitions
        filled_labels = np.array([-1]*len(censoring_mask))
        filled_labels[censoring_mask] = labels
    else:
        filled_labels = labels

    if exclude_steadystate:
        unsteady_filled_labels = [filled_labels[0]]
        i=0
        while(i+1<len(filled_labels)):
            if filled_labels[i]!=filled_labels[i+1]:
                unsteady_filled_labels.append(filled_labels[i+1])
            i+=1

        filled_labels = np.array(unsteady_filled_labels)
    
    transitions_str = np.array([
        f"{x}_{y}_{z}" for x, y, z in zip(filled_labels[:-2], filled_labels[1:-1], filled_labels[2:])
    ])

    k = labels.max()+1
    transition_counts=np.zeros([k,k*k])
    for x in range(k):
        for y in range(k):
            for z in range(k):
                # rows are the origin CAP, and column the subsequent CAP
                transition_counts[x,((y*k)+z)]=(transitions_str==f"{x}_{y}_{z}").sum()
    # divide rows by their sum to get transition probability from a CAP to the target
    # i.e. given a certain CAP state, what is the probability of the next CAP
    transition_probability = (transition_counts.T/transition_counts.sum(axis=1)).T
    return transition_counts, transition_probability