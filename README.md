# HPC
Repository for CSC-592, the general goal of the project is to use machine learning to accurately predict the outcome of NBA games.

## Dropout Network
I believe this is a working example of a dropout vs. non-dropout network. As it stands, dropping nodes results in slightly lower accuracy (<0.5%), but the loss of the network is much elss (~6%-8%). I'm not 100% sure what loss is, but it appears that loss is what the network uses to correct itself and update weights. A lower value for this statistic means our network is able to learn faster. In theory, this means our dropout network should be able to learn more quickly (and easily) parameters thrown at it.

### Results
While the results of the dropout vs non-dropout networks are fairly similar, this is most likely because a large number of input parameters are not given for the dropout network. In our project, can have a large array of statistics as input nodes.
