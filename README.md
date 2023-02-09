


## Brief description of models
NMTFcoclust and ELBMcoclust implement co-clustering algorithms with two percpective matrix factorization and latent block model

- $NMTF_{\alpha}$
```math
	D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
```

- $ONMTF_{\alpha}$
```math
	D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
	+
	\delta \; Tr(\mathbf{R}\Psi_{g}\mathbf{R}^{\top})
	+
	\beta \;  Tr(\mathbf{C}\Psi_{s}\mathbf{C}^{\top}),
```


## Co-clustering

### Example of Co-clustering on Document-Word (Term) Matrix 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Topic_detection_in_a_document-word_matrix.gif?raw=true" width="45%">




### Word Cloud Co-clustering for [Digikala Persian Comments](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digikala%20comments) 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/ezgif.com-gif-maker.gif?raw=true" width="70%">


### Word Cloud Co-clustering for [Persian News](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news)
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Persian_nesw.gif?raw=true" width="70%">


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



```python
import pickle
                                                                   # Read Data Sets ------->  Digikala
# Loading pickle data from a file
with open('tfidf_Digikala.pkl', 'rb') as f:
        tfidf_Digikala = pickle.load(f)

# Loading pickle data from a file
with open('labels_Digikala', 'rb') as f:
        labels_Digikala = pickle.load(f)

true_labels = np.sort(labels_Digikala)
```

## Model
```python
from NMTFcoclust.Models.NMTFcoclust_OPNMTF_alpha import OPNMTF
from NMTFcoclust.Models.NMTFcoclust_ONMTF_alpha import ONMTF
from NMTFcoclust.Models.NMTFcoclust_NMTF_alpha import NMTF
from ELBMcoclust.Models.coclust_ELBMcem import CoclustELBMcem
from ELBMcoclust.Models.coclust_SELBMcem import CoclustSELBMcem
```
```python
ONMTF_alpha = ONMTF(n_row_clusters = 3, n_col_clusters = 3, delta = 0.03,  beta = 0.03,  alpha = 0.1, max_iter=1)
ONMTF_alpha.fit(tfidf_Digikala)

NMTF_alpha = NMTF(n_row_clusters = 3, n_col_clusters = 3, alpha = 2, max_iter=1)
NMTF_alpha.fit(tfidf_Digikala)

from sklearn.metrics import confusion_matrix 

confusion_matrix(np.sort(true_labels), np.sort(ONMTF_alpha.row_labels_))

from NMTFcoclust.Evaluation.EV import Process_EV

Process_Ev = Process_EV( np.sort(true_labels), tfidf_Digikala , ONMTF_alpha) 
Process_Ev = Process_EV( np.sort(true_labels), tfidf_Digikala , NMTF_alpha) 



Accuracy (Acc):0.8761116222017786
Normalized Mutual Info (NMI):0.6836524406477642
Adjusted Rand Index (ARI):0.7667679710034221
Confusion matrix   (CM):
[[2181  201    0]
 [   0  216  203]
 [   0    0  460]]
```

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Wordcloud_Digimag_33.png?raw=true" width="70%">

Word Cloud Co-clustering for [Persian News](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news)

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Results/Persian%20news/Wordcloud_8_8_Persian_news.png?raw=true" width="100%">


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
