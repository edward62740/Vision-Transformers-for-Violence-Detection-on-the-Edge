# Models

This folder contains the tflite/torch model definitions and trained weights

In the case of the proposed modifications in section 2.1.1 - 2.1.4, they are found in deit.py and constructed with reconstruct_deit.py.<br>
The output .onnx is converted to tf format (Colab - onnx2tf.ipynb) and is ready to use for training/forward pass of features.

Note that the ops are not compatible with tflite micro.