"""
    trainStep.function : "executes" the "optimizer" node that was created 
                         in step 3 within a tensorflow session.
"""

import tensorflow as tf 

def function(session, optimizer):
    """Runs the optimizer node for current graph

    Args:
        session: current tf session
        optimizer: (Tensor) graph node representing optimizer operation

    Returns:
        none
    """

    session.run(optimizer)