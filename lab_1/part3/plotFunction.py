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
    breakpoint()
    plt.plot(lossValues[:,0,0])
    plt.show()
