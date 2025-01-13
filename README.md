# Autoencoder Neural Network for Feature Extraction

## Introduction
In this project, an autoencoder neural network with a single hidden layer is implemented for unsupervised feature extraction from natural images. The following cost function is minimized:

$$
J_{\text{ae}} = \frac{1}{N} \sum_{n=1}^N \|w(o) - o(d)\|^2 + \lambda \sum_{i=1}^{H} \sum_{j=1}^{H} \|W_{ij}\|^2 + \beta \sum_{i=1}^{H} \text{KL}(\rho \| \hat{\rho})
$$

The terms in the cost function are defined as follows:
- The first term is the average squared error between the desired response and the network output across training samples. Note that the desired output is the same as the input.
- The second term enforces Tikhonov regularization on the connection weights with parameter $\lambda$.
- The last term enforces that the hidden unit activations are sparse with parameter $\beta$, controlling the relative weighting of this term. The level of sparsity is controlled via the Kullback-Leibler divergence (KL-divergence) between a Bernoulli variable with mean $\rho$ and another term $\hat{\rho}_i$, which is the average activation of hidden unit $i$ across training samples.

## Tasks

### a) Data Preprocessing
The file `data` contains a collection of 16x16 RGB patches extracted from various natural images. The data preprocessing steps are as follows:
1. Convert the images to grayscale using the luminosity model:
   
   Y = 0.21R + 0.72G + 0.07B
   
2. Normalize the data by removing the mean pixel intensity of each image and dividing by its standard deviation.
3. To visualize the data, randomly sample patches in RGB format and separately display the normalized versions of the same patches. Comment on your results.

### b) Weight Initialization
Prior to training, initialize the weights and the bias terms as uniform random numbers in the interval $[w_{\text{min}}, w_{\text{max}}]$, where:

$$
w_{\text{min}} = -\sqrt{\frac{6}{L_\text{in} + L_\text{out}}}, \quad w_{\text{max}} = \sqrt{\frac{6}{L_\text{in} + L_\text{out}}}
$$

Here, $L_\text{in}$ and $L_\text{out}$ are the number of input and output units, respectively.

Divide the data into training, validation, and test sets in a structured format. The following fields should be used: `trainX`, `trainY`, `valX`, `valY`, `testX`, and `testY`. Use 80% of the data for training, 10% for validation, and 10% for testing. Implement the autoencoder and train it to reconstruct the input data while minimizing the cost function.

### c) Network Performance
The following outputs are required:
- Display the first layer of connection weights as grayscale images.
- Analyze the effect of changes in the network parameters ($\lambda$, $\beta$, $\rho$) on the reconstruction performance.

### d) Testing with Different Sparsity Levels
Train the network for three different values of $\rho$ (low, medium, high) at $\lambda = 0.01$ and $\beta = 0.1$. Display the hidden-layer activations as grayscale images and compare the results for different combinations of training parameters.
