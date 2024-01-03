# Violence Detection 
 
This project aims to design efficient neural networks for use in detecting human violence and terrorism through edge-deployable surveillence devices.<br>
Specifically, the Vision Transformer (ViT), and more generally attention (MHSA) modules are used in the proposed design.

## Background
Typically, image and video-recognition tasks use CNN, due to properties such as (spatial) translation invariance. Transformer architecture first saw major success in NLP applications.
ViTs were first introduced in [ref: an image is with 16x16 words], outperforming CNNs but requiring significantly more training. Recent improvements of ViTs have allowed for reduction of the quadratic time complexity of MHSA and training on smaller datasets etc.,
by using hybrid architectures and distillation respectively. There now exist many smaller ViTs, comparable to small CNNs such as MobileNet. <br>


## Overview
In this project, the architecture of the model is as follows:

[image]

<br>

Transfer learning on imagenet1/21k is used for the spatial feature extractor, which takes in an image $\mathbb{R}^{H \times W \times C}$. Various different ViTs (and CNNs) for reference are used.
The output features $\mathbb{R}^{1 \times V}$ are collected over N timesteps and fed into the temporal segment as $\mathbb{R}^{1 \times N \times V}$, which utilizes a transformer encoder architecture.

To realize higher performance, the proposed model is then fully quantized to int8 to be run on the Edge TPU accelerator. PTQ is used for this approach - it is likely that a model trained from scratch with QAT will have higher accuracy.
Full integer quantization causes some problems for the ViT models. Specifically, this includes the values in the attention map, which have very large outliers which cases minmax quantization to be inaccurate, due to large bin size. A quantization scheme like log2
can be used to reduce this problem. 
<br>
$$Q(X|b) = \text{sign}(X) \cdot \text{clip}\left(\lfloor - \log_2 \frac{|X|}{\max(|X|)} \rceil, 0, 2^{b-1} - 1\right) \quad \quad \quad \quad (1)$$
where $clip(\cdot)$ specifies the lower and upper bounds respectively that constrain the quantized value.<br> <br>
Next, the use of non-linear activation functions in transformers, such as GeLU (due to the lack of support for gaussian erf in this case), also is present in ViT, which is not supported by the TPU. Approximations are therefore used at the cost of less accuracy, given by:
$$GELU(x) = 0.5x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right]\right) \quad \quad \quad \quad (2)$$

The following tests (and training) were conducted on a subset of the UCF-Crime dataset, which contains surveillence videos of crime events. Specifically, the classes uses were (Arson, Assault, Explosion, Fighting, Robbery, Shooting), as they all correspond to violent behaviors (i.e. shoplifting is not necessarily violent).
The labelled data size was augmented with another dataset UCFCrime2Local, which provides spatio-temporal labels for some of the videos.
To address the lack of inductive bias of transformer, random cropping/rotation was used during training.
In general, the SGD optimizer was found to converge to more optimal solutions than Adam or RMSProp etc., for training the transformer encoder (temporal segment).

|                               | AUC/Accuracy (float32) | AUC/Accuracy (int8) | AUC/Accuracy (TPU) | 1-sequence latency (ms) | Parameters (M) |
|-------------------------------|------------------------|---------------------|--------------------|-------------------------|----------------|
| ResNet50 + Bidirectional LSTM | 0.8564/0.8508          |                     | Not supported      |                         | 25             |
| EfficientNet v2 + Transformer | 0.8745/0.8527          | 0.8429/0.8301       | Same               | <60ms                   | 8.1+2.2        |
| DeiT-T + Transformer          | 0.877                  | 0.850               | Same               |                         | 5.5+1.3        |

_1-sequence latency is defined as the time taken for a single forward pass of image through the entire spatio-temporal network_
