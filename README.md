# Vision Transformers for Violence Detection on the Edge

This project uses Vision Transformers (ViT) as backbone networks for video violence detection on the edge.<br>
<br>
Various pre-trained ViTs and hybrid ViTs are used as backbone, yielding on average 2-3% better accuracy than CNN/LSTM baseline methods.<br>
Additionally, this project also aims to deploy the violence detection model onto the Google Edge TPU. As such, this work proposed various techniques to modify the ViT graph structure for execution on TPU.<br>
Due to limitations on quantization schemes for hybrid ViTs, the DeiT model was used for deployment on TPU.<br>

The table below provides an overview of the codes used. Each subdirectory contains a README to describe the usage.
| Folder | Description                                          |
|--------|------------------------------------------------------|
| [models](https://github.com/edward62740/Vision-Transformers-for-Violence-Detection-on-the-Edge/tree/main/models) | Tflite/torch model definitions and trained weights   |
| [test](https://github.com/edward62740/Vision-Transformers-for-Violence-Detection-on-the-Edge/tree/main/test)   | Testing code for measuring accuracy etc.             |
| [train](https://github.com/edward62740/Vision-Transformers-for-Violence-Detection-on-the-Edge/tree/main/train)  | Training code                                        |
| [utils](https://github.com/edward62740/Vision-Transformers-for-Violence-Detection-on-the-Edge/tree/main/utils)  | Preprocessing, QAT, PTQ, image processing algorithms |

This project was completed under NTU's URECA programme. Refer to this [link]() for the paper.

The proposed modifications in section 2.1.1 - 2.1.4 are found in models/deit.py and constructed with models/reconstruct_deit.py.<br>
The TFlite model (and the one compiled for TPU) is in models/deit+transformer.<br>
The UCF-crime dataset is available [here](https://www.crcv.ucf.edu/projects/real-world/).<br>


