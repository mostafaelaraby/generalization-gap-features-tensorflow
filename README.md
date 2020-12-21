
<br />
<p align="center">

  <h3 align="center">Feature extractors for PGDL competition</h3>

  <p align="center">
    Tensorflow 2.0 features extractor used in Neurips competiton PGDL 2020.
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#references">References</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Our objective in PGDL is to be able to predict the generalization gap of artificial neural networks in a robust manner. For that purpose, i have created this repo to extract state-of-the-art features that are used for generalization gap prediction. The generalization gap is defined as the difference between the network accuracy on the training set and the network accuracy on the validation (or test) set. 
A list of related papers that I find helpful are listed in the references.

### Built With

The extractors are built with Python3 and the following main libraries
* [Tensorflow](https://www.tensorflow.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)



<!-- GETTING STARTED -->
## Getting Started
After cloning this repository, some prerequisites needs to be installed and PGDL start kit needs to be downloaded

### Prerequisites

Install required libraries listed in requirements.txt file
  ``` 
    pip3 install -r requirements.txt
  ```

### Installation

1. Clone this repo
2. Download PGDl start kit from [Codalab](https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit)
3. Add path to "ingestion_program" from the start kit in extract_meta_features.py line 7


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Mostafa Elaraby - [@melaraby91](https://twitter.com/melaraby91)


<!-- REFERENCES -->
## References
* [1]P. Bartlett, D. J. Foster, and M. Telgarsky, “Spectrally-normalized margin bounds for neural networks,” arXiv:1706.08498 [cs, stat], Dec. 2017, Accessed: Oct. 21, 2020. [Online]. Available: http://arxiv.org/abs/1706.08498.
* [2]Y. Jiang, D. Krishnan, H. Mobahi, and S. Bengio, “Predicting the Generalization Gap in Deep Networks with Margin Distributions,” arXiv:1810.00113 [cs, stat], Jun. 2019, Accessed: Jul. 27, 2020. [Online]. Available: http://arxiv.org/abs/1810.00113.
* [3]T. Unterthiner, D. Keysers, S. Gelly, O. Bousquet, and I. Tolstikhin, “Predicting Neural Network Accuracy from Weights,” arXiv:2002.11448 [cs, stat], May 2020, Accessed: Sep. 27, 2020. [Online]. Available: http://arxiv.org/abs/2002.11448.
* [4]S. Yak, J. Gonzalvo, and H. Mazzawi, “Towards Task and Architecture-Independent Generalization Gap Predictors,” Jun. 2019, Accessed: Oct. 02, 2020. [Online]. Available: https://arxiv.org/abs/1906.01550v1.
* [5]A. Virmaux and K. Scaman, “Lipschitz regularity of deep neural networks: analysis and efficient estimation,” in Advances in Neural Information Processing Systems 31, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds. Curran Associates, Inc., 2018, pp. 3835–3844



