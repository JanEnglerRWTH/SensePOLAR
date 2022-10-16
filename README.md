# POLAR_C
The first (semi-) supervised framework for augmenting interpretability into contextual word embeddings (BERT).
Interpretability is added by rating words on scales that encode user-selected senses, like correctness or left-right-direction.

## Usage
This first commit only contains the most basic functionalities of the POLAR_C framework.
Any given word will be transformed into our interpretable Word Embedding space and the individual ratings on the scales can be analyzed.
The scales, where the word is rated the highest (we call these the 'top POLAR_C dimensions'), are shown.
These dimensions are usually the most descriptive dimensions for a word.

After installing the python packages below:
Simply run 'python polarC.py' in the main folder and follow the instructions.

## Prerequisites
### Python packages
* scipy 1.7.1
* transformers 4.12.3
* pickleshare 0.7.5
* torch 1.9.0
* torchaudio 0.9.0
* torchmetrics 0.5.1
* torchvision 0.10.0
* numpy 1.21.1
* datasets 2.28.1
* sklearn 1.1.2

* (python 3.8.10)



## Literatur
Static POLAR framework:
Mathew, B., Sikdar, S., Lemmerich, F., & Strohmaier, M. (2020, April). The polar framework: Polar opposites enable interpretability of pre-trained word embeddings. In Proceedings of the Web Conference 2020 (pp. 1548-1558).
BERT Model:
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
