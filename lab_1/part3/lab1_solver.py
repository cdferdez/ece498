from __future__ import print_function
import numpy as np
import random
import sys
from functools import reduce
import tensorflow as tf
import initializeX
import lossFunction
import optimizerFunction
import computeLoss
import trainStep
# import plotFunction

# Preliminary setup, do not modify
if len(sys.argv) > 1:
    random.seed(int(sys.argv[1]));
    np.random.seed(int(sys.argv[1]));
else:
    random.seed(int(1));
    np.random.seed(int(1));

def shape(V):
    return list(map(int, list(V.shape)));

total = lambda shape : reduce(lambda x, y : x * y, shape, 1);
assert(total([3,4,1]) == 12);
# Preliminary setup over

"""
In this experiment, we are solving for a value of X that results in a(X^t*X) + b^t*X = y (* represents 
matrix multiply, and ^t represents transpose). Here X, b are a column-vectors of shape [4,1], a
and y are scalar constants. We will use gradient descend to perform this task. 
We setup the problem as follows: we compute Z = a*(X^t*X) + b^t*X, and then
the quantity (Z - y)**2. This quantity is called the 'loss' in our problem setting. Note that loss
is higher when Z deviates from y, and lower otherwise. We will use tensorflow to minimize the value of the loss by
adjusting the value of X 'appropriately'. This is done by repetitively computing the
gradient of the loss with respect to X, and adjusting X using this gradient.

You will implement the following functions for this problem
1. initializeX.function : function returning the tensorflow variable X initialized to a random value.
   Provide a file initializeX.py with "function" implemented within it

2. lossFunction.function : function implementing the loss, (Z - y) ** 2.
   Provide a file lossFunction.py with "function" implemented within it

3. optimizerFunction.function : function implementing the training step
   Provide a file optimizerFunction.py with "function" implemented within it

4. computeLoss.function : function that provides a printable value of the loss. By printable, what we mean
   is that the value of loss is visible on using python print(). Using print directly on tensorflow variables
   or constants doesn't show their value.
   Provide a file computeLoss.py with "function" implemented within it

5. trainStep.function : function that implements the training step
   Provide a file trainStep.py with "function" implemented within it

"""

#1. Create the constant a
a = tf.constant(10, dtype=tf.float32);

#2. Create the variable X. Here, recommended that you initialize X 
# from a numpy array with random numbers selected from between 0 and 1.
X = initializeX.function(shape=(4,1));

shapeOfX = shape(X);
if not ((len(shapeOfX) == 2) and (shapeOfX[0] == 4) and (shapeOfX[1] == 1)):
    raise ValueError("Variable X doesn't have the correct shape");
else:
    print("X of correct shape has been returned");

#3. Create the constant b
b = tf.constant(np.arange(4).reshape((4,1)), dtype=tf.float32);

#3. Create the constant y
y = tf.constant(15, dtype=tf.float32);

#4. Create the tensorflow computation graph. The graph outputs 
# the loss function that was described above. The function you
# write must evaluate (a(X^t*X) + b^t*X - y) ** 2
loss = lossFunction.function(a, X, b, y);

shapeOfLoss = shape(loss);
print("Loss has shape", shapeOfLoss);
assert(total(shapeOfLoss) == 1), "Loss is not a scalar!";

#5. Create the AdamOptimizer. Optimizers add additional nodes 
# in the tensorflow graph to compute gradients as well as apply them to the variables involved.
# This could be manually performed using tf.gradients etc, but it is a process that is repeated
# over and over in all Deep Neural Networks, so the optimizers hide all the gory details.
# In addition, optimizers do things other than calculate simple gradients in order to ensure
# that convergence happens quickly. "Executing" the optimizer inside a tf.Session hence implements
# 1) computation of gradients with respect to all the variables in the graph, and 2) adjusting the value
# of the variables using these gradients
optimizer = optimizerFunction.function(loss, lr=1e-3);

#6. Launch the training loop. We want to track the loss function over the training iterations
# We will launch 250 training iterations
session = tf.Session();
session.run(tf.global_variables_initializer());

lossValues = [];

for i in range(250):
    # 7. Implement a function that provides the printable value of loss
    lossValue = computeLoss.function(session, loss);

    # 8. Implement a function that performs loss minimization
    trainStep.function(session, optimizer);

    # 9. Print out loss
    print("Iteration %d, loss = %f"%(i, lossValue));

    lossValues.append(lossValue);

### Please note that the following is not for demo, but only for the report (hence, currently commented out)
# You may prepare the plot using another tool like Excel, but this is the recommended way.
# # 10. Finally, add a function to plot the loss value across training steps
# # Please refer to the python library, pyplot https://matplotlib.org/users/pyplot_tutorial.html
# plotFunction.function(lossValues);
