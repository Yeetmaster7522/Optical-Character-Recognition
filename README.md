# Software AT3
## Due Term 2 Week 11, Wednesday 1st July 2026

Goal is to create a good CNN to perform optical character recognition. ***Running on Python 3.12.3 in a virtual environment***. Execution does not require a virtual environment however it is suggested.

---

### Dependencies

Requires PyTorch, Torchvision, Matplotlib, Numpy, and Pillow.

> Commands below will install all dependencies:
>
> - pip install torch torchvision
> - pip install matplotlib
>
> To install PyTorch with GPU support visit [this page](https://pytorch.org/get-started/locally/) for more details

---

### Model accuracy (validation set with noise) as of 10/06/2, 00:00:
> #### Trained on images without noise
>
> - model_1_F_23: *94%*
> - model_2_F_54: *96%*
> - model_3_F_66: *96%*
> - model_4_F_43: *95%*
>
> #### Trained on images with noise
>
> - model_1_T_52: *97%*
> - model_2_T_69: *98%*
> - model_3_T_36: *98%*
> - model_4_T_39: *97%*


NOTES
Get google library fonts
make dataloader close images with pillow
