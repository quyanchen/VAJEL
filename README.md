# VAJEL

This is the code for our paper: Protein Complex Identification via Joint Embedding Learning with a Variational Graph Autoencoder.

## **Requirements**

- tensorflow  1.15.0
- sklearn  0.24.2
- numpy  1.19.5
- python  3.6.13

Install all dependencies using

```python
pip install -r requirements.txt
```


## **Datasets**

| Dataset | Proteins | Edges |  Cliques | Average Neighbors |
| --- | --- | --- | --- | --- |
| Biogrid | 5640 | 59748 | 38616 | 21.187 |
| DIP | 4928 | 17201 | 7832 | 6.981 |
| Krogan14K | 3581 | 14076 | 4075 | 7.861 |

## **Usage**

### PreProcessing

Python 1_construct_knowledege-enhanced_ppi.py to Construct Knowledge-Enhanced_PPI Network

### Variational Graph Autoencoder-Based Joint Learning

Python 2_train_VAJEL.py to achieve Variational Autoencoder-Based Joint Learning

### Core-Attachment Clustering Algorithm

Python 3_Evaluation.py to identify protein complexes and to evaluate performance.
