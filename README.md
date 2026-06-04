[![License](https://img.shields.io/badge/License-BSD%203--Clause-black.svg)](LICENSE)
[![https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Models/NMTFcoclust_ONMTF_alpha.py](https://badgen.net/badge/ONMTF/Coclust/black?icon=instgrame)](https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Models/NMTFcoclust_ONMTF_alpha.py)
![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Persian/Coclust/black?icon=instgrame)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Persian/Text/black?icon=instgrame)](https://github.com/Saeidhoseinipour/Persiancoclust/tree/master/Datasets)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Persian/Wordcloud/black?icon=instgrame)](https://github.com/Saeidhoseinipour/Persiancoclust/tree/master/Datasets)





# Table of Contents
1. [Notation](#notation)
2. [Objective Functions](#objective-functions)
3. [Co-clustering](#co-clustering)
   - [NMTF](#nmtf)
   - [Example of Co-clustering on Word-Document Matrix](#example-of-co-clustering-on-word-document-matrix)
   - [Word Cloud Co-clustering for Digikala Persian Comments](#word-cloud-co-clustering-for-digikala-persian-comments)
4. [Datasets](#datasets)
5. [Model](#model)
6. [Visualization](#visualization)
7. [Cite](#cite)
8. [References](#references)

## Notation

- $\mathbf{X}\mathbf{X}$: Word-Document counts, Movie-Viewer ratings, Product-Customer purchases matrices 

- \mathbf{R}\mathbf{R}: Row-coefficient matrix

- \mathbf{B}\mathbf{B}: Block  matrix

- \mathbf{C}\mathbf{C}: Column-coefficient matrix



## Objective functions

- NMTF_{\alpha}NMTF_{\alpha}
```math
	D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
```

- ONMTF_{\alpha}ONMTF_{\alpha}
```math
	D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
	+
	\delta \; Tr(\mathbf{R}\Psi_{g}\mathbf{R}^{\top})
	+
	\beta \;  Tr(\mathbf{C}\Psi_{s}\mathbf{C}^{\top}),
```


## Co-clustering

### NMTF

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/NMTF.png?raw=true" width="100%">


### Example of Co-clustering on Word-Document Matrix 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Topic_detection_in_a_document-word_matrix.gif?raw=true" width="100%">




### Word Cloud Co-clustering for [Digikala Persian Comments](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digikala%20comments) 
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/ezgif.com-gif-maker.gif?raw=true" width="100%">




## Datasets

| Datasets | Documents | Words  | Number of clusters |
| -- | ----------- | --  | -- |
| [Digikala](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digikala%20comments) |3261  |10728  |3  |
| [Digimag](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Digimag) | 852 | 80160 | 7 |
| [Persian news](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news) |1644  |28216  |8  |
| [Psychological advice](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Psychological%20advise%20text%20in%20Persian) | 79 |  1929|11  |
| [Snappfood](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Snappfood) |7000  |8735   |2  |
| [Political websites](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Political%20websites) |229  |102390   |4  |
| [Varzesh3](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/varzesh3) |54  |4303   |3  |

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Datasets/all_datasets_combined_2x3.png?raw=true" width="100%">

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
from NMTFcoclust.Models.NMTFcoclust_ONMTF_alpha import ONMTF
from NMTFcoclust.Models.NMTFcoclust_NMTF_alpha import NMTF
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
Confusion Matrix   (CM):
[[2181  201    0]
 [   0  216  203]
 [   0    0  460]]
```


<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Wordcloud_Digimag_33.png?raw=true" width="100%">

Word Cloud Co-clustering for [Persian News](https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news)

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Results/Persian%20news/Wordcloud_8_8_Persian_news.png?raw=true" width="100%">


## Cite
Please cite the following paper in your publication if you are using [`Persiancoclust`]() in your research:

```bibtex
 @article{Persiancoclust, 
    title={Penalized Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Persian Text Co-clustering}, 
    DOI={Preprint}, 
    journal={Iranian Journal of Science (preprint)}, 
    authors={Saeid Hoseinipour, Adel Mohammadpour}, 
    year={2026}
} 
```



<!--
##  Presentation video

[![Presentation video for OPNMTF, Text mining, Matrix factorization, Co-clustering](https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/OPNMTF_video.png)](https://www.youtube.com/watch?v=LCamkfTYGyM&t=5s)


<p align="center">
  <a href="https://www.youtube.com/watch?v=LCamkfTYGyM&t=5s">
    <img src="https://github.com/Saeidhoseinipour/NMTFcoclust/blob/master/Doc/Image/OPNMTF_video.png" alt="Presentation video for OPNMTF, Text mining, Matrix factorization, Co-clustering" style="width:60%; transform: perspective(1000px) rotateY(-70deg);">
  </a>
</p>


<p align="center">
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C14.09 3.81 15.76 3 17.5 3 20.58 3 23 5.42 23 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
  </svg>
</p>


<a href="">
  <img src="" alt="Presentation video for OPNMTF, Text mining, Matrix factorization, Co-clustering, Saeid Hoseinipour" style="width: 70%;">
</a>
-->




## References

[1] [Mehrdad Farahani et al, Parsbert: Transformer-based model for Persian language understanding, Neural Processing Letters (2021).](https://github.com/Saeidhoseinipour/parsbert) 

[2] [Yoo et al, Orthogonal nonnegative matrix tri-factorization for co-clustering: Multiplicative updates on Stiefel manifolds (2010), 
	Information Processing and Management.](https://www.sciencedirect.com/science/article/abs/pii/S0306457310000038)
	
[3] [Ding et al, Orthogonal nonnegative matrix tri-factorizations for clustering, Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2008).](https://dl.acm.org/doi/abs/10.1145/1150402.1150420)

[4] [Long et al, Co-clustering by block value decomposition, Proceedings of the Eleventh ACM SIGKDD International Conference on Knowledge Discovery in Data 	Mining (2005).](https://dl.acm.org/doi/abs/10.1145/1081870.1081949)

[5] [Li et al, Nonnegative Matrix Factorization on Orthogonal Subspace (2010), Pattern Recognition Letters.](sciencedirect.com/science/article/abs/pii/S0167865509003651)

[6] [Cichocki et al, Non-negative matrix factorization with $\alpha$-divergence (2008), Pattern Recognition Letters.](https://www.sciencedirect.com/science/article/abs/pii/S0167865508000767)

[7] [Hoseinipour et al, Orthogonal Parametric Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Co-clustering (2023), Expert Systems With Application.](https://doi.org/10.1016/j.eswa.2023.120680)

[8] [Hoseinipour et al, Orthogonal parametric non-negative matrix tri-factorization with $\alpha$-Divergence for co-clustering, *Expert Systems with Applications* (2023).](https://doi.org/10.1016/j.eswa.2023.120680)

[9] [Hoseinipour et al, A Sparse Exponential Family Latent Block Model for Co-clustering (2025), *Advances in Data Analysis and Classification* (2024).](https://doi.org/10.1007/s11634-024-00608-3)

