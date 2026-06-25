# Software AT3
## Due Term 2 Week 11, Wednesday 1st July 2026

Goal is to create a good CNN to perform optical character recognition. ***Running on Python 3.12.3 in a virtual environment***. Execution does not require a virtual environment however it is suggested.

---

### Dependencies

Requires PyTorch, Torchvision, scikit-learn, Matplotlib, Numpy, and Pillow.

> Commands below will install all dependencies:
>
> - pip install torch torchvision
> - pip install matplotlib
> - pip install pandas
> - pip install scikit-learn
>
> To install PyTorch with GPU support visit [this page](https://pytorch.org/get-started/locally/) for more details

---

### Model accuracy (validation set with noise) as of 20/06/26, 07:26:
> #### Trained on images without noise
>
> - model_1_F_23: *94.51%*
> - model_2_F_54: *96.67%*
> - model_3_F_66: *96.72%*
> - model_4_F_43: *96.50%*
>
> #### Trained on images with noise
>
> - ocr_v1_t0.034_v0.047.pt: *98.55%*
> - ocr_v2_t0.025_v0.032.pt: *98.81%*
> - ocr_v3_t0.027_v0.030.pt: *98.81%*
> - ocr_v4_t0.033_v0.027.pt: *98.90%*

---
# Note to self
Need to hyperoptimise training to make it as fast as possible. Add progress bars. Force limit on how many models are saved per training run. Add entire google font library for fun. Do wingdings for maximum pain and suffering >:). Add accuracy and confidence per font.