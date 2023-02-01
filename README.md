


## Brief description of models
NMTFcoclust and ELBMcoclust implement co-clustering algorithms with two percpective matrix factorization and latent block model
- $OPNMTF_{\alpha}$ 
```math
D_{\alpha}(\mathbf{X}||\mathbf{RBC}^{\top})+
  \lambda \; D_{\alpha}(\mathbf{I}_{g}||\mathbf{R}^{\top}\mathbf{R})+
  \mu \; D_{\alpha}(\mathbf{I}_{s}||\mathbf{C}^{\top}\mathbf{C})
```
- $ELBMcem$
```math
   \sum\limits_{k} r_{.k} \log\pi_{k} +	
	\sum\limits_{h}
	c_{.h}\log\rho_{h}  +
	Tr\left(
	(\mathbf{R}^{\top} (\mathbf{S_{x}}\odot \hat{\boldsymbol{\beta}}) \mathbf{C})^{\top}
	\mathbf{A}_{\boldsymbol{\alpha}}
	\right)- 
	Tr\left(
	(\mathbf{R}^{\top} (\mathbf{E}_{mn}\odot
	\hat{\boldsymbol{\beta}}) \mathbf{C})^{\top}
	\mathbf{F}_{\boldsymbol{\alpha}}
	\right)
```
## Co-clustering

### Document-Word Matrix
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Topic_detection_in_a_document-word_matrix.gif?raw=true" width="45%">

### Word Cloud Co-clustering
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/ezgif.com-gif-maker.gif?raw=true" width="45%">



## Requirements
```python
numpy==1.18.3
pandas==1.0.3
scipy==1.4.1
matplotlib==3.0.3
scikit-learn==0.22.2.post1
coclust==0.2.1

```
## Installing NMTFcoclust

## License


## Datasets

| Datasets | Documents | Words | Sporsity | Number of clusters |
| -- | ----------- | -- | -- | -- |
| [Digimag](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digimag) | 6896 | 80160 | 96% | 7 |
| [Digikala](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digikala%20comments) |3261  |10728  | 91.83% |3  |
| [Snappfood](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Snappfood) |3891  |4303  |98%  |3  |
| [Psychological advice text Persian](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Psychological%20advise%20text%20in%20Persian) | 79 |  1929| 99.99% |11  |
| [Persian news](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news) |4069  |18483  | 99.99% |5  |


```python
import pandas as pd 
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix 


                                                                   # Read Data Sets ------->  Classic3

file_name=r"D:\My paper\Application\NMTFcoclust\Dataset\Classic3\classic3.mat"
mydata = loadmat(file_name)

                                                                    # Data matrix 
X_Classic3 = mydata['A'].toarray()
X_Classic3_sum_1 = X_Classic3/X_Classic3.sum()
                                                                   
true_labels = mydata['labels'].flatten().tolist()                   # True labels list [0,0,0,..,1,1,1,..,2,2,2]  n_row_cluster = 3
true_labels = [x+1 for x in true_labels]                            # True labels list [1,1,1,..,2,2,2,..,3,3,3]  n_row_cluster = 3
print(confusion_matrix(true_labels, true_labels))


```

## Model
```python
from NMTFcoclust.Models import NMTFcoclust_ONMTF_alpha

```
```python
from NMTFcoclust.Models.NMTFcoclust_OPNMTF_alpha_2 import OPNMTF
from NMTFcoclust.Evaluation.EV import Process_EV

OPNMTF_alpha = OPNMTF(n_row_clusters = 3, n_col_clusters = 3, landa = 0.3,  mu = 0.3,  alpha = 0.4, max_iter=1)
OPNMTF_alpha.fit(X_Classic3_sum_1)
Process_Ev = Process_EV( true_labels ,X_Classic3_sum_1, OPNMTF_alpha) 



Accuracy (Acc):0.9100488306347982
Normalized Mutual Info (NMI):0.7703948803438703
Adjusted Rand Index (ARI):0.7641161476685447
Adjusted Mutual Info (AMI):0.7702867787943636
Intra-cluster Average Similarity (IAS):0.027380015679156534
Inter-cluster Centroids Average Similarity (ICAS):0.335635399782488
Runtime:4.049925799999983
Confusion matrix   (CM):
[[1033    0    0]
 [ 276 1184    0]
 [   0   74 1324]]
Total Time:  26.558243700000276
```

![DC](https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Wordcloud_Digimag_33.png?raw=true)

## Cite
Please cite the following paper in your publication if you are using [Persian-Textmining]() in your research:

```bibtex
 @article{Persianmining, 
    title={Textual summarization of persian corpus by co-clustering algorithms.}, 
    DOI={Preprint}, 
    journal={preprint}, 
    author={Saeid Hoseinipour, Mina Aminghafari, Adel Mohammadpour}, 
    year={2023}
} 
```
## References

[1] [Mehrdad Farahani et al, Parsbert: Transformer-based model for Persian language understanding, Neural Processing Letters (2021).](https://github.com/Saeidhoseinipour/parsbert) 


