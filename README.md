# SparseBERT

This repository provides a script and recipe to search the BERT model with sparse attention mask for PyTorch to balance the performance and efficiency.
Our implementation is an further version of the [NVIDIA implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT), which leverages mixed precision arithmetic and Tensor Cores on Volta V100 and Ampere A100 GPUs for faster training times while maintaining target accuracy.
 
 
 
## Model overview
 
BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. 

### Model architecture
 
The BERT model uses the same architecture as the encoder of the Transformer. Input sequences are projected into an embedding space before being fed into the encoder structure. Additionally, positional and segment encodings are added to the embeddings to preserve positional information. The encoder structure is simply a stack of Transformer blocks, which consist of a multi-head attention layer followed by successive stages of feed-forward networks and layer normalization. The multi-head attention layer accomplishes self-attention on multiple input representations.
 
An illustration of the architecture taken from the [Transformer paper](https://arxiv.org/pdf/1706.03762.pdf) is shown below.
 
 ![BERT](images/model.png)
 
 In this work, we just focus on BERT-base with following configuration.
 
 | **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERT-base |12 encoder| 768| 12|4 x  768|512|110M|
 
 
## Quick Start Guide
 
Our main difference is `run_pretraining.py` and `modeling.py`.
For quick start guide, you can refer to NVIDIA's repository.

To run on a single node 8 x V100 32G cards, from within the container, you can use the following script to run pre-training.  
`bash scripts/run_pretraining.sh`

The default hyperparameters are set to run on 8x V100 32G cards.

|  <img width=200/> |**Sparsity ratio**|**MNLI&#8209;(m/mm)**| **QQP** | **QNLI**|  **SST&#8209;2** | **COLA** |  **STS&#8209;B**| **MRPC**| **RTE**| **Average**|
|:--------|:--------:|:----------:|:----:|:---:|:--------:|:---:|:----:|:----:|:----:|:----:|
|Strided [1]|     70.4 | 81.9/81.9 | 87.1 | 89.0 | 91.7 | 58.4 | 86.6 | 86.1 | 52.7 | 79.5 |
|Fixed  [1] |    72.7 | 81.4/81.8 | 86.4 | 88.1 | 91.3 | 54.2 | 85.9 | 88.7 | 59.2 | 79.7|
|Longformer [2]|  88.7 | 80.5/81.0 | 86.8 | 88.4 | 91.8 | 57.9 | 86.9 | 81.7 | 65.3 | 80.1 |
|LogSparse [3] | 89.8 | 77.9/78.2 | 85.9 | 84.6 | 92.0 | 58.5 | 83.2 | 82.0 | 58.8 | 77.9 |
|BigBird [4]   | 93.2 | 80.2/80.1 | 86.4 | 87.6 | 91.6 | 54.3 | 84.7 | 84.1 | 66.0 | 79.4 |
|Star   [5]   | 96.1 | 79.1/79.0 | 86.2 | 86.4 | 91.2 | 59.6 | 84.7 | 83.9 | 60.3 | 78.9 |
|DAM<sub>u</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-4}">)   | 78.9 | 82.2/82.6 | 87.3 | 89.7 | 92.4 | 57.3 | 86.5 | 89.2 | 70.8 | 82.0|
|DAM<sub>u</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-3}">)   | 79.2 | 82.2/82.4 | 87.1 | 89.5 | 92.3 | 57.2 | 86.2 | 89.1 | 67.9 | 81.6 |
|DAM<sub>u</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-2}">)   | 79.8 | 81.7/82.3 | 86.8 | 89.4 | 92.1 | 57.2 | 86.1 | 89.0 | 67.1 | 81.3 |
|DAM<sub>u</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-1}">)   | 85.8 | 81.4/82.2 | 86.5 | 89.1 | 92.1 | 56.6 | 84.4 | 88.3 | 66.8 | 80.8 |
|DAM<sub>s</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-4}">)   | 91.2 | 81.7/81.7 | 87.0 | 88.3 | 92.5 | 59.4 | 86.7 | 88.4 | 63.2 | 80.9 |
|DAM<sub>s</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-3}">)   | 91.6 | 81.0/81.2 | 86.9 | 88.0 | 92.4 | 58.6 | 86.2 | 85.7 | 62.8 | 80.3 |
|DAM<sub>s</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-2}">)   | 91.7 | 81.1/80.9 | 86.9 | 87.9 | 92.3 | 57.9 | 84.8 | 85.4 | 61.0 | 79.8 |
|DAM<sub>s</sub> (<img src="https://render.githubusercontent.com/render/math?math=\lambda=10^{-1}">)   | 93.5 | 80.9/81.0 | 86.7 | 87.7 | 92.2 | 57.7 | 84.8 | 85.2 | 59.9 | 79.6 |

[1] [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf)  
[2] [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf)  
[3] [Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://papers.nips.cc/paper/2019/file/6775a0635c302542da2c32aa19d86be0-Paper.pdf)  
[4] [Big Bird: Transformers for Longer Sequences](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)  
[5] [Star-Transformer](https://www.aclweb.org/anthology/N19-1133.pdf)  
