#ISMwithLSTM.jl
For this project we tried to reproduce the results presented in the [paper](https://iopscience.iop.org/article/10.1088/1748-9326/ac0acb/meta) by Takahito Mitsui and Niklas Boers who used neural networks (or more particularly echo state networks) in order to predict the onset date of the Indian Summer Monsoon about two to three months in advance. This prediction is relevant for agricultural planning and water-resource management in India. The echo state networks in this case were trained on temperature data.

In our work we attempted to reproduce their results by replacing the echo state network used by Mitsui and Boers by a long short-term memory (LSTM) network and otherwise using the exact same approach.
