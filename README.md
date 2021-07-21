# cvDNN-runner
OpenCV DNN profiler

`main.cpp` runs inference on a set of available backends.
Available parameters: 
* model name;
* input mat size;
* batch size;
* type of input format (0 - NHWC, 1 – NCHW);
* backend type (0 - CPU, 1 – GPU/CUDA, 2 – ALL);
* media path (video file / camera device index).
