
from importlib.metadata import distribution
from utils import get_dataset

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np



def display_instances(batch):
    """
    This function takes a batch from the dataset and display the image with 
    the associated bounding boxes.
    """
    # color mapping of classes
    colormap = {1: [1, 0, 0], 2: [0, 1, 0], 4: [0, 0, 1]}
    
    # create an empty plot
    f, ax = plt.subplots(3, 4, figsize=(20, 10))
    
    # plot each sample
    for i, sample in enumerate(batch):
        x = i % 3
        y = i % 4
        img = sample["image"].numpy()
        ax[x, y].imshow(img)
        
        bboxes = sample['groundtruth_boxes'].numpy()
        classes = sample['groundtruth_classes'].numpy()
        
        for cl, bb in zip(classes, bboxes):
            y1 , y2 = bb[[0,2]] * img.shape[0]
            x1, x2 = bb[[1,3]] * img.shape[1]
            rec = Rectangle((x1, y1), x2- x1, y2-y1, facecolor='none', 
                            edgecolor=colormap[cl])
            ax[x, y].add_patch(rec)
        ax[x ,y].axis('off')
    plt.tight_layout()
    plt.show()
    

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

def display_distribution(dataset):
    """
    This function takes the dataset and display the data distribution 
    """
    class_distribution = np.array([])
    object_distribution = np.array([])
    
    for sample in dataset:
        classes = sample['groundtruth_classes'].numpy()
        class_distribution = np.append(class_distribution, classes )
        object_distribution = np.append(object_distribution, len(classes))
    unique, counts = np.unique(class_distribution, return_counts=True)
    
    distribution_fig, distribution_ax = plt.subplots()
    explode = (0.1, 0.1, 0.1)
    labels = 'Vehicles', 'Pedestrians', 'Cyclists'
    distribution_ax.pie(counts, explode=explode, labels=labels, autopct=make_autopct(counts))
    distribution_ax.axis('equal') 
    
    plt.show()
    
    hist_fig, hist_ax = plt.subplots()
    hist_ax.set(xlabel='Number of Lables in Image', ylabel='Number of Images')
    hist_ax.hist(object_distribution, bins=10, linewidth=0.5, edgecolor="white")
    
    plt.show()




if __name__ == "__main__": 
    dataset = get_dataset("./data/train/*.tfrecord")
    instances_sample_size = 12
    instances_batch = dataset.shuffle(instances_sample_size, reshuffle_each_iteration=True).take(instances_sample_size)
    display_instances(instances_batch)
    
    distribution_sample_size = 1000
    distribution_batch = dataset.shuffle(distribution_sample_size, reshuffle_each_iteration=True).take(distribution_sample_size)
    display_distribution(distribution_batch)