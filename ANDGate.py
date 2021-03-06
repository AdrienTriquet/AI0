import numpy as np

#############################################################################################################################################################################################################
#############################################################################################################################################################################################################



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
def activationFunction(n):

    #TODO - Application 1 - Step 4b - Define the binary step function as activation function
    #return (n >= 0) * 1
    return 1.0/(1.0 + np.exp(-n))
 #############################################################################################################################################################################################################
#############################################################################################################################################################################################################

#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
def forwardPropagation(p, weights, bias):

    # TODO - Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias   =>  n

    n = p[0] * weights[0] + p[1] * weights[1] + bias

    # TODO - Application 1 - Step 4c - Pass the result to the activation function  =>  a

    a = activationFunction(n)

    return a
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
def main():

    #Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    #The network should receive as input two values (0 or 1) and should predict the target output


    #Input data
    P = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

    #Labels
    # for AND
    #t = [0, 0, 0, 1]

    # for OR
    t = [0, 1, 1, 1]

    #TODO - Application 1 - Step 2 - Initialize the weights with zero  (weights)
    weights = [0, 0]

    #TODO - Application 1 - Step 2 - Initialize the bias with zero  (bias)
    bias = 0


    #TODO - Application 1 - Step 3 - Set the number of training steps  (epochs)
    """ Seeking the epochs number : we see that 100 and 10 gives the same.
    By testing from 0 to 10, I saw that I had valid prediction at 5"""
    epochs = 5


    #TODO - Application 1 - Step 4 - Perform the neuron training for multiple epochs
    for ep in range(epochs):
        for i in range(len(t)):

            #TODO - Application 1 - Step 4 - Call the forwardPropagation method
            a = forwardPropagation(P[i], weights, bias)


            #TODO - Application 5 - Compute the prediction error (error)
            error = t[i] - a


            #TODO - Application 6 - Update the weights
            weights[0] = weights[0] + error * P[i][0]
            weights[1] = weights[1] + error * P[i][1]


            #TODO - Update the bias
            bias = bias + error


    #TODO - Application 1 - Print weights and bias
    print("Weights = " + str(weights))
    print("Bias = " + str(bias))


    # TODO - Application 1 - Step 7 - Display the results
    for i in range(len(t)):
        predict = forwardPropagation(P[i], weights, bias)
        print("Input = " + str(P[i]) + " - " + "Prediction = " + str(predict) + " - " + "Actual label = " + str(t[i]))

    return predict
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
if __name__ == "__main__":
    main()
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
