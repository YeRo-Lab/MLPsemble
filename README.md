Overview:
MLPsemble is an ensemble neural network that utilizes Logistic Regression, K-Nearest Neighbors, and Random Forests
as base models and uses Out-Of-Fold predictions to train a Multi-Layer Perceptron. Its purpose is to identify rare
retinal ganglion cells without needing large amounts of memory or runtime.


To run MLPsemble_tutorial.py:

1. Clone this github using the link https://github.com/YeRo-Lab/MLPsemble.git or download the .zip file 
2. To use the Tran dataset run the join_tran.py file and ensure the output points to the directory 'Data/Tran/'
3. Go to the main() in MLPsemble and ensure that the file name matches with RuiChen.h5ad, Rheaume.h5ad, or Tran.h5ad
4. Next run MLPsemble.py to see the tutorial on how the model works


Access to the original publications containing the datasets can be found below:

Rheaume - https://www.nature.com/articles/s41467-018-05134-3
Tran - https://www.cell.com/neuron/fulltext/S0896-6273(19)30969-9
Rui Chen - https://www.nature.com/articles/s41588-025-02454-1

