[![License](https://img.shields.io/badge/License-BSD%203--Clause-black.svg)](LICENSE)
[![https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Models/NMTFcoclust_ONMTF_alpha.py](https://badgen.net/badge/ONMTF/Coclust/black?icon=instgrame)](https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Models/NMTFcoclust_ONMTF_alpha.py)
![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Persian/Coclust/black?icon=instgrame)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Persian/Text/black?icon=instgrame)](https://github.com/Saeidhoseinipour/Persiancoclust/tree/master/Datasets)
[![https://github.com/Saeidhoseinipour/NMTFcoclust](https://badgen.net/badge/Persian/Wordcloud/black?icon=instgrame)](https://github.com/Saeidhoseinipour/Persiancoclust/tree/master/Datasets)

# **`Persiancoclust`**
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Image/IJS.png?raw=true" width="96%">

Official implementation of the paper:

**Penalized   Non-negative Matrix Tri-Factorization with $\alpha$-Divergence for Persian Text Co-clustering**

`Saeid Hoseinipour`, `Adel Mohammadpour`
  
Published in **Iranian Journal of Science**, 2026. 

DOI: https://doi.org/10.1007/s11634-024-00608-3  
Code repository: https://github.com/Saeidhoseinipour/Persiancoclust  
Download: [![](https://badgen.net/badge/Orginal/Paper/black?icon=instgrame)](https://link.springer.com/article/10.1007/s11634-024-00608-3#rightslink)  
<img src="https://raw.githubusercontent.com/Saeidhoseinipour/ELBMcoclust/main/Images/selfhst--gmail.svg" width="16" height="16" alt="email" style="vertical-align: middle; margin-right: 4px;"> **Call for paper** — [saeidhoseinipour9@gmail.com](mailto:saeidhoseinipour9@gmail.com)

## Links

- Paper DOI: https://doi.org/10.1007/s11634-024-00608-3
- Code DOI: https://doi.org/10.5281/zenodo.21093419
- Official GitHub repository: https://github.com/Saeidhoseinipour/ELBMcoclust
- Supplementary material: https://github.com/Saeidhoseinipour/EM-typecoclust/tree/main
- Datasets: https://github.com/Saeidhoseinipour/ELBMcoclust/tree/main/Datasets
- Visualizations: https://github.com/Saeidhoseinipour/ELBMcoclust/tree/main/Images


## Table of Contents
<table>
  <tr>
    <td style="vertical-align: top;">
      <ul>
        <li><a href="#notation">Notation</a></li>
        <li><a href="#objective-functions">Objective Functions</a></li>
        <li><a href="#co-clustering">Co-clustering</a></li>
        <ul>
          <li><a href="#nmtf">NMTF</a></li>
          <li><a href="#example-of-co-clustering-on-word-document-matrix">Example of Co-clustering on Word-Document Matrix</a></li>
          <li><a href="#word-cloud-co-clustering-for-digikala-persian-comments">Word Cloud Co-clustering for Digikala Persian Comments</a></li>
        </ul>
        <li><a href="#datasets">Datasets</a></li>
        <li><a href="#model">Model</a></li>
        <li><a href="#visualization">Visualization</a></li>
        <li><a href="#cite">Cite</a></li>
        <li><a href="#references">References</a></li>
      </ul>
    </td>
    <td>
      <img alt="Co-clustering on word–document matrix by Saeid Hoseinipour"
           src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Topic_detection_in_a_document-word_matrix.gif?raw=true"
           style="width:230px;">
    </td>
    <td>
      <img alt="Screenshot: 'README.md'"
           src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/ezgif.com-gif-maker.gif?raw=true"
           style="width:190px;">
    </td>
  </tr>
</table>


## Notation

- **X**: Word-Document counts, Movie-Viewer ratings, Product-Customer purchases matrices 

- **R**: **R**ow-coefficient matrix

- **B**: **B**lock  matrix

- **C**: **C**olumn-coefficient matrix




## Objective functions

- NMTF
	```math
		D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
	```

- PNMTF
	```math
		D_{\alpha}(\mathbf{X}|| \mathbf{RBC}^{\top})
		+
		\delta \; Tr(\mathbf{R}\Psi_{g}\mathbf{R}^{\top})
		+
		\beta \;  Tr(\mathbf{C}\Psi_{s}\mathbf{C}^{\top}),
	```


## Co-clustering

### NMTF

<p align="center">
  <img alt="Co-clustering on document-word matrix" 
  src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/NMTF.png?raw=true" 
  width="100%">
</p>

<p align="center">
  <em>
  The input document–word matrix X ∈ R⁺<sup>n×m</sup> is factorized as X ≈ R B C<sup>T</sup>, 
  where R encodes document memberships (row clusters), C encodes word memberships (column clusters), 
  and B summarizes the strength of association between document and word clusters. The reordered matrix 
  reveals co-clusters as block structures, where rows are first permuted (word grouping) and columns are 
  then permuted (document grouping), making latent topics visually apparent.
  </em>
</p>


### Example of Co-clustering on Word-Document Matrix 
<img alt="Co-clustering on word–document matrix by Saeid Hoseinipour" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Topic_detection_in_a_document-word_matrix.gif?raw=true" width="100%">

* Co-clustering of a word–document matrix. Rows (words) are first permuted to group terms with similar distributions across documents, followed by a permutation of columns (documents) to cluster similar texts. The resulting block-diagonal structure highlights coherent word–document co-clusters.*



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


**For more details see [this page](https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Datasets/Readme.md)**


- **Digikala**: User comments are included in a dataset provided by the Open Data Mining Program (ODMP). This dataset comprises 3261 comments, each labeled as ‘No Idea’, ‘Not Recommended’, and ‘Recommended’.

- **Digimag**: This collection has 852 articles from the online magazine Digikala. There are 7 different types of articles: ‘Video Games’, ‘Shopping Guide’, ‘Health Beauty’, ‘Science Technology’, ‘General’, ‘Art Cinema’, and ‘Books Literature’.

- **Persian news**: A collection of news posts collected from various Persian online news sources. The total number of documents is 1644 and is divided into 8 categories: ‘Social’, ‘Economic’, ‘International’, ‘Political’, ‘Science Technology’, ‘Cultural Art’, ‘Sport’, and ‘Medical’.

- **Psychological advice**: 79 people's questions about personal problems classified into 11 psychological topics such as: ‘Obsession’, ‘Anxiety’, ‘Shyness’, ‘Distraction’, ‘Addiction’, ‘Family’, ‘Courtship’, ‘Guilt’, ‘Jealousy’, ‘Relaxation’, and ‘Study’.

- **Snappfood**: This dataset contains 7000 user comments collected from Snappfood (an online food-delivery platform in Iran). The comments are labeled for binary sentiment polarity as ‘Happy’ or ‘Sad’.

- **Political websites**: Using an API request on 5 February 2026, this text-based dataset was collected from 229 Persian news websites, focusing on political topics. It comprises 102,390 words and categorizes the websites into 4 clusters: ‘Principlist’, ‘Reformist’, ‘Independent’, and ‘Opposition’.


<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Datasets/all_datasets_combined_2x3.png?raw=true" width="100%">
*Frequency of documents per category across the six Persian datasets, where labels represent the true document categories assigned to each dataset.*



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


<p align="center">
  <b style="font-size:40px;">
    Reorganized word cloud Co-clustering for 
    <a href="https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Persian%20news">
      Persian News
    </a>
  </b>
</p>

<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Results/Persian%20news/Wordcloud_8_8_Persian_news.png?raw=true" width="100%">



<p align="center">
  <b style="font-size:100px;">
# Sensitivity of the parameter α on six datasets analyzed using three evaluation measures.
  </b>
</p>
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Image/Comparison_All_Datasets.svg?raw=true" width="100%">


<p align="center">
  <b style="font-size:40px;">
Sensitivity of the parameters δ and β on six datasets analyzed by accuracy.  </b>
</p>
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Image/Sensitivity_Delta_Beta_Combined.jpg?raw=true" width="100%">




<p align="center">
  <b style="font-size:40px;">
    Reorganized bar charts for co-clustering of 
    <a href="https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Political%20websites">
      Political websites
    </a>
  </b>
</p>



<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persian-Textmining/blob/master/Image/Total_BarCharts.jpg?raw=true" width="100%">



<p align="center">
  <b style="font-size:40px;">
    Reorganized word clouds for co-clustering of 
    <a href="https://github.com/Saeidhoseinipour/Persian-Textmining/tree/master/Datasets/Political%20websites">
      Political websites
    </a>
  </b>
</p>
<img alt="Screenshot: 'README.md'" src="https://github.com/Saeidhoseinipour/Persiancoclust/blob/master/Image/Total_WordClouds.jpg?raw=true" width="100%">







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

[7] [Hoseinipour et al, Orthogonal parametric non-negative matrix tri-factorization with $\alpha$-Divergence for co-clustering, Expert Systems with Applications (2023).](https://doi.org/10.1016/j.eswa.2023.120680)

[8] [Hoseinipour et al, A Sparse Exponential Family Latent Block Model for Co-clustering, Advances in Data Analysis and Classification (2024).](https://doi.org/10.1007/s11634-024-00608-3)

