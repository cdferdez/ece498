""" 
    plotFunction.function(lossValues)
"""
import numpy as np 
import matplotlib.pyplot as plt

def function(lossValues):
    """ 
        Plots the loss values vs training steps

        Args:
            lossValues (list): loss values from loss minimization
    """
    # convert to array, shape was kinda weird as a list
    lossValues = np.array(lossValues)
    
    plt.plot(lossValues[:,0,0])
    plt.title("Loss Function vs Iterations | Christian Fernandez CDF2")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Values")
    plt.xlim(-5,255)
    plt.show()
