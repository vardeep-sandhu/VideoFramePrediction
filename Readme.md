
<!-- ABOUT THE PROJECT -->

## About The Project 

Video Frame Prediciton is the task in computer vision which consists of providing a model with a sequence of past frames, and asking it to generate the next frames in the sequence (also referred to as the future frames). Despite the fact that humans can easily and effortlessly solve the future frame prediction problem, it is extremely challenging for a machine. The fact that the model needs to understanf the physical dynamics of motion in real world makes this task complex for machines. This task has many downstream applications in autonomoous driving like predicting future state of agents, thus detecting moving objects from a sequence. Here we present our approach which explots Convolutional Layers for feature encoding and decoding, Convolutional LSTMs for predicting future frames of Moving MNIST and KTH dataset. A detailed description of algorithms and analysis of the results are available in the [Report](). 
<!-- Add pdf link here -->



### Built With
This project was built with 

* python v3.8.5
* PyTorch v1.7
* The environment used for developing this project is available at [environment.yml](environment.yml).


## TODO

- [ ] Add flag to specify after how many epochs should a model be stored


<!-- GETTING STARTED -->

## Getting Started

Clone the repository into a local machine and enter the [src](src) directory using

```shell
git clone https://github.com/here-to-learn0/Video_frame_prediction
cd Video_frame_prediction/
```

### Prerequisites

Create a new conda environment and install all the libraries by running the following command

```shell
conda env create -f environment.yml
```

The datasets used in this project (Moving MNIST and KTH) will be automatically downloaded and setup in `data` directory during execution.



### Instructions to run

To train the model specify the dataset config file to use with the `-c` flag. 

```sh
python scripts/main.py -c mnist.yaml 
```


This trains the frame prediction model and saves it in the `model` directory.

This generates folders in the `results` directory for every log frequency steps. The folders contains the ground truth and predicted frames for the train dataset and test dataset. These outputs along with loss and metric are written to Tensorboard as well.




## Model overview

The architecture of the model is shown below. The frame predictor model takes in the first ten frames as input and predicts the future ten frames. The
discriminator model tries to classify between the true future frames and predicted future frames. For the first ten time instances, we use the ground truth past frames as input, where as for the future time instances, we use the past predicted frames as input.

![Transformer](./docs/readme/stconv.jpg)



<!-- RESULTS -->

## Results

Detailed results and inferences are available in report [here](./docs/report.pdf).

We evaluate the performance of the model for long-term predictions to reveal its generalization capabilities. We provide the first 20 frames as input and let the model predict for the next 100 frames. 

Ground truth frames (1-10):

![Transformer](./docs/readme/results_1.png)

Predicted frames (2-101):

![Transformer](./docs/readme/results_11.png)



We evaluate the performance of the model on out-of-domain inputs which the model has not seen during the training. We provide a frame sequence with one moving digit as input and observe the outputs from the model.

Ground truth frames (1-10):

![Transformer](./docs/readme/results_6.png)

Predicted frames (2-41):

![Transformer](./docs/readme/results_61.png)



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->

## Contact

Vineeth S - vs96codes@gmail.com

Project Link: [https://github.com/vineeths96/Video-Frame-Prediction](https://github.com/vineeths96/Video-Frame-Prediction)



## Acknowledgments

> Base code is taken from:
>
> https://github.com/JaMesLiMers/Frame_Video_Prediction_Pytorch


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- 
[contributors-shield]: https://img.shields.io/github/contributors/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[contributors-url]: https://github.com/vineeths96/Video-Frame-Prediction/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[forks-url]: https://github.com/vineeths96/Video-Frame-Prediction/network/members
[stars-shield]: https://img.shields.io/github/stars/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[stars-url]: https://github.com/vineeths96/Video-Frame-Prediction/stargazers
[issues-shield]: https://img.shields.io/github/issues/vineeths96/Video-Frame-Prediction.svg?style=flat-square
[issues-url]: https://github.com/vineeths96/Video-Frame-Prediction/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/vineeths96/Video-Frame-Prediction/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/vineeths -->
