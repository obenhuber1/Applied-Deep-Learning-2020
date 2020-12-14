# Applied-Deep-Learning-2020
TU course on Deep Learning



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![LinkedIn][linkedin-shield]][linkedin-url ]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/obenhuber1/Applied-Deep-Learning-2020">
    <img src="res/simba_title.jpg" alt="Logo" width="200">
  </a>

  <h3 align="center">Is it Simba ?</h3>

  <p align="center">
    Image Classification for three classes using CNN
    <br />
    <a href="https://github.com/obenhuber1/Applied-Deep-Learning-2020/tree/main/doc"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
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
        <li><a href="#replicating-the-toolchain">Replicating the Toolchain</a></li>
        <li><a href="#running-the-model">Running the Model</a></li>
      </ul>
    </li>
        <li>
      <a href="#final-web-application">Final Web Application</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
My project is inspired by my dog Simba, a three year old Golden Retriever. I built a Convolutional Neural Network (VGG3 Style) which tries to identify if the dog on a given image is Simba, a Golden Retriever (not being Simba) or some other dog breed (multiclass problem with 3 classes).

The Dataset used for training consists of ~ 1000 pictures from Simba taken by me between 1.3.2019 and 14.12.2020 as well as the same number of images for the other two classes from the <a href="http://vision.stanford.edu/aditya86/ImageNetDogs/main.html" target="_blank">"Stanford Dog Dataset"</a> and the <a href="https://cg.cs.tsinghua.edu.cn/ThuDogs/" target="_blank">"Tsinghua Dog Dataset"</a>.

Implementation was done in Python using Package "Keras" (not tf.keras) and Tensorflow.

Later on here comes a screenshot of the final application.
[![Product Name Screen Shot][product-screenshot]](https://example.com)

### Built With
* Python 3.6.10
* Tensorflow GPU 2.1.0
* Keras 2.3.1
* Additional Packages: os, sys, numpy, matplotlib, sklearn
* IDE Spyder



<!-- GETTING STARTED -->
## Getting Started

Either use your existing Toolchain and customize it or create a fresh Environment to replicate the Toolchain (below steps were tested on Anaconda3 Powershell for Windows10).

Please make sure you have an up-and-running Python environment with the following packages:
* Tensorflow GPU 2.1.0
* Keras 2.3.1
* Additional Packages: os, sys, numpy, matplotlib, scikit-learn

### Replicating the Toolchain
```
conda create -n simba python=3.6 tensorflow-gpu
conda activate simba
pip install keras==2.3.1
conda install matplotlib
conda install scikit-learn
conda install spyder
spyder

```


### Running the Model

1. Download and Extract Python Scripts and Dataset: https://github.com/obenhuber1/Applied-Deep-Learning-2020/blob/main/test/release_ex2/github_simba_ex2.zip

2. Open 'simba_cnn_vgg3_data_prep.py' and adapt the "Main Directory" in line 11

3. Run Data Preparation Script 'simba_cnn_vgg3_data_prep.py'. ATTENTION: it will delete the directory contained in variable 'dst_directory' including all subfolders
   Check if subfolders under /images_model/ where populated properly
 
4. Run Model Training Script 'simba_cnn_vgg3_model.py' and check results in folder /model_outputs/

5. In case you want to do some predictions using your own data put them in the folder /images_model/test and run the Script 'simba_cnn_vgg3_predict.py'


<!-- APPLICATION -->
## Running the Web App

To get a local copy up and running follow these simple steps:

### Prerequisites

Please make sure you have an up-and-running Python environment with the following packages:
* ...


### Installation

1. ...

2. ...

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License.



<!-- CONTACT -->
## Contact

Christoph Obenhuber - christoph.obenhuber@gmx.at



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[license-url]: https://github.com/obenhuber1/Applied-Deep-Learning-2020/tree/main/res/LICENSE.txt
[linkedin-url]: https://www.linkedin.com/in/christoph-obenhuber-2752564/
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
