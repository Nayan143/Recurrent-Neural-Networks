# Recurrent-Neural-Networks

A basic RNN (Recurrent Neural Network) implemented from the scratch by providing guidance for the key steps in the process, namely forward propagation and backward propagation through time (BPTT). This will be followed by a test case, where the network is trained to remember an arbitrary given sequence across several time steps, and outputting the same sequence after reading a specific symbol. This kind of application tests the memory capabilities of the RNN, as well as the ability to learn the concept of first remembering and then repeating.

Requirements: The following packages are required: numpy, tensorflow and matplotlib


# main.py
- Forward propagation and backward propagation through time
- Exploding gradients and gradient clipping
- Memorization task
- Run training


# BasicRNNCell.py
- Class RNN cell (contains the implementation of basic RNN cell)
  - Forward propagation of basic RNN (LinearLayer.py, BasicRNNCell.__init__ <-- variable name)
- BPTT of basic RNN
- Fix the Vanishing gradient


# common.py
- softmax
- cross entrophy loss
- Exploding gradient (check clip_gradient function)


# LinearLayer.py :
- Glorot initialization of weights and zero initialization for derivatives
- gradients clipping (clip_gradients func)
- weights updating (update_weights func)

# train.py :
- Run the training routine on the network specified as a list of layers in net on data and labels
- Iterate over all layers of the RNN
- Compute the column-wise softmax of the output
- Compute cross entropy loss
- Compute gradients from cross entropy loss and softmax
- Backprop into network
- run_train_memory
- Create random training data
- Run the accuracy test for the trained model


# data_generator.py :
- Generate the data for the memory test
- Generate a random dataset for usage in visualizing exploding gradients


# test.py
- run exploding gradients test
- testing exploding gradients
- run memory test



# Memorization Task : 
It's needed that our RNN to learn to perform a simple memorization task. The goal is to take an input sequence and read until a specific character is encountered. After this point the RNN should output all the characters that were previously read in the correct order. To this end it's required to implement the function run_memory_test which computes the accuracy of the network on a randomly generated test set of data. For data generation you may want to use the function generate_data_memory, which returns data and label matrices. Additionally, the function run_forward_pass may be helpful to you in order to obtain the soflty assigned classes. That the only interesting values for the accuracy measure are the ones generated after the delimiter has been read. Afterwards you may try to adjust the command line parameters (e.g.learning rate, number of steps) so that the test accuracy is maximized.

# Gradient clipping : 
As a method for mitigation of exploding gradients, gradient clipping was introduced before. This method computes the norm of the gradient after BPTT and scales it to a fixed value (run_exploding_grads_test to make sure that the gradient clipping has taken effect correctly)

# Exploding gradient: 
One big flaw of recurrent networks, is that the gradient of parameter matrices may vanish or become extremely large (use the function run_exploding_grads_test to visualize how quickly the mean absolute value of the gradient can increase. You may also try out different numbers for the hidden layer, sequence length and input vocabulary to see how this affects the magnitude of the gradient)




