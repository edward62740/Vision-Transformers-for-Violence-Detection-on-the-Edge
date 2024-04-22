# Utils

This folder contains the preprocessing, QAT, PTQ, image processing algorithms.

| Folder        | Description                                                                     |
|---------------|---------------------------------------------------------------------------------|
| preprocessing | Converts the dataset into valid training data                                   |
| quantization  | PTQ/QAT code as well as code to measure the performance of the quantized models |
| tpu-compile   | (Colab) For executing the edgetpu compiler and other graph modifications        |
<br>

For preprocessing, the general steps are as follows:
- conv_video_to_img.py converts each .mp4 video into a sequence of downsampled frames
- conv_sp_annotations.py converts the dataset's annotations into a suitable numpy array format
- conv_img_to_npy.py converts the images from conv_video_to_img.py output to numpy arrays
- forward_pass_extr.py does a forward pass of the outputs of conv_img_to_npy.py through the chosen FE
  
<br>
After training (for PTQ), the tflite_conversion.py and tflite_conversion_temporal.py are run on the FE and transformer encoder/LSTM respectively to give the quantized models.
<br>
The models can then be co-compiled with compile.ipynb.