


## Brief description of models
NMTFcoclust and ELBMcoclust implement co-clustering algorithms with two percpective matrix factorization and latent block model
- $ONMTF_{\alpha}$
```math
	D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
	+
	\delta \; Tr(\mathbf{R}\Psi_{g}\mathbf{R}^{\top})
	+
	\beta \;  Tr(\mathbf{C}\Psi_{s}\mathbf{C}^{\top}),
```
- $NMTF_{\alpha}$
```math
	D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
```
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

### Example of Co-clustering on Document-Word (Term) Matrix 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Topic_detection_in_a_document-word_matrix.gif?raw=true" width="45%">

### Word Cloud Co-clustering for [Digikala Persian Comments](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digikala%20comments) 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/ezgif.com-gif-maker.gif?raw=true" width="45%">


### Word Cloud Co-clustering for [Persian News](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news)
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Persian_nesw.gif?raw=true" width="45%">

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
| [Digikala](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digikala%20comments) |3261  |10728  | 91.83% |3  |
| [Digimag](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digimag) | 6896 | 80160 | 96% | 7 |
| [Persian news](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news) |1644  |28216  | 99.99% |8  |
| [Psychological advice text Persian](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Psychological%20advise%20text%20in%20Persian) | 79 |  1929| 99.99% |11  |
| [Snappfood](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Snappfood) |3891  |4303  |98%  |3  |

**For more details see [this page](https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Datasets/Readme.md)**

|          Dataset         | Documents   | Words       |  Sporsity |Unbalance| Number of clusters |
|:------------------------:|:-----------:|:-----------:|:-----------:|:-----:|:-------------:|
|  Digikala                |      -      |    81.72    |    81.74*   | 80.74 |       -       |
|  Digimag                 |      -      |    87.98    |    88.12*   | 87.87 |       -       |
|  Persian news            |      -      |    71.31*   |    71.11    |   -   |     69.33     |
| Psychological advice     |      -      |    92.42*   |    92.13    |   -   |     91.98     |


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
from NMTFcoclust.Models.NMTFcoclust_ONMTF_alpha import OPNMTF
from ELBMcoclust.Models.coclust_ELBMcem import CoclustELBMcem
from ELBMcoclust.Models.coclust_SELBMcem import CoclustSELBMcem
```
```python
ONMTF_alpha = ONMTF(n_row_clusters = 3, n_col_clusters = 3, delta = 0.03,  beta = 0.03,  alpha = 0.1, max_iter=1)
ONMTF_alpha.fit(tfidf_Digikala)

NMTF_alpha = ONMTF(n_row_clusters = 3, n_col_clusters = 3, alpha = 2, max_iter=1)
NMTF_alpha.fit(tfidf_Digikala)

from sklearn.metrics import confusion_matrix 

confusion_matrix(np.sort(true_labels), np.sort(ONMTF_alpha.row_labels_))

from NMTFcoclust.Evaluation.EV import Process_EV

Process_Ev = Process_EV( np.sort(true_labels), tfidf_Digikala , ONMTF_alpha) 
Process_Ev = Process_EV( np.sort(true_labels), tfidf_Digikala , NMTF_alpha) 



Accuracy (Acc):0.8761116222017786
Normalized Mutual Info (NMI):0.6836524406477642
Adjusted Rand Index (ARI):0.7667679710034221
Adjusted Mutual Info (AMI):0.6834128837244977
Silhouette score :0.0007204093021165255
Calinski harabasz score:2.8469395382993867
Davies bouldin score :24.506823121591225
Intra-cluster Average Similarity (IAS):0.040789570962311075
Inter-cluster Centroids Average Similarity (ICAS):0.91052293820662
Runtime:'0:00:05.079835'
Confusion matrix   (CM):
[[2181  201    0]
 [   0  216  203]
 [   0    0  460]]
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

[2] Saeid et al, Orthogonal parametric matrix tri-factorization with $\alpha$-divergence for co-clustering (2023), Preprint.

[3] Saeid et al, Sparse Expoential family latent block model for co-clustering (2023), Preprint.
