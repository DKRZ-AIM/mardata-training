import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction(image, prob):
    prob = prob.data.numpy().squeeze()
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(image, cmap='gray')
    
    ax[1].barh(np.arange(10), prob)
    ax[1].set_aspect(0.1)
    ax[1].set_yticks(np.arange(10))
    ax[1].set_yticklabels(np.arange(10))
    ax[1].set_title('Probability of Class')
    ax[1].set_xlim(0, 1.1)
    plt.tight_layout()