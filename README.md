\documentclass[a4paper,12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{float}

\begin{document}

\title{Autoencoder Neural Network for Feature Extraction}
\author{}
\date{}
\maketitle

\section*{Introduction}
In this project, an autoencoder neural network with a single hidden layer is implemented for unsupervised feature extraction from natural images. The following cost function is minimized:

\begin{equation}
J_{\text{ae}} = \frac{1}{N} \sum_{n=1}^N \|w(o) - o(d)\|^2 + \lambda \sum_{i=1}^{H} \sum_{j=1}^{H} \|W_{ij}\|^2 + \beta \sum_{i=1}^{H} \text{KL}(\rho \| \hat{\rho})
\end{equation}

The terms in the cost function are defined as follows:
\begin{itemize}
    \item The first term is the average squared error between the desired response and the network output across training samples. Note that the desired output is the same as the input.
    \item The second term enforces Tikhonov regularization on the connection weights with parameter $\lambda$.
    \item The last term enforces that the hidden unit activations are sparse with parameter $\beta$, controlling the relative weighting of this term. The level of sparsity is controlled via the Kullback-Leibler divergence (KL-divergence) between a Bernoulli variable with mean $\rho$ and another term $\hat{\rho}_i$, which is the average activation of hidden unit $i$ across training samples.
\end{itemize}

\section*{Tasks}
\subsection*{a) Data Preprocessing}
The file \texttt{assign3\_data1.mat} contains a collection of 16x16 RGB patches extracted from various natural images. The data preprocessing steps are as follows:
\begin{itemize}
    \item Convert the images to grayscale using the luminosity model:
    \begin{equation}
    Y = 0.21R + 0.72G + 0.07B
    \end{equation}
    \item Normalize the data by removing the mean pixel intensity of each image and dividing by its standard deviation.
    \item To visualize the data, randomly sample patches in RGB format and separately display the normalized versions of the same patches. Comment on your results.
\end{itemize}

\subsection*{b) Weight Initialization}
Prior to training, initialize the weights and the bias terms as uniform random numbers in the interval $[w_{\text{min}}, w_{\text{max}}]$, where $w_{\text{min}} = -\sqrt{6/(L_\text{in} + L_\text{out})}$ and $w_{\text{max}} = \sqrt{6/(L_\text{in} + L_\text{out})}$. Here, $L_\text{in}$ and $L_\text{out}$ are the number of input and output units, respectively. 

Divide the data into training, validation, and test sets in a structured format. The following fields should be used: \texttt{trainX}, \texttt{trainY}, \texttt{valX}, \texttt{valY}, \texttt{testX}, and \texttt{testY}. Use $80\%$ of the data for training, $10\%$ for validation, and $10\%$ for testing. Implement the autoencoder and train it to reconstruct the input data while minimizing the cost function.

\subsection*{c) Network Performance}
The following outputs are required:
\begin{itemize}
    \item Display the first layer of connection weights as grayscale images.
    \item Analyze the effect of changes in the network parameters ($\lambda$, $\beta$, $\rho$) on the reconstruction performance.
\end{itemize}

\subsection*{d) Testing with Different Sparsity Levels}
Train the network for three different values of $\rho$ (low, medium, high) at $\lambda = 0.01$ and $\beta = 0.1$. Display the hidden-layer activations as grayscale images and compare the results for different combinations of training parameters.

\end{document}
