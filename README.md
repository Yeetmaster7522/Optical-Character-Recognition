# Software AT3
## Due Term 2 Week 11, Wednesday 1st July 2026

Goal is to create a good CNN to perform optical character recognition. ***Running on Python 3.12.3 in a virtual environment***. Execution does not require a virtual environment however it is suggested.

---

### Dependencies

Requires PyTorch, Torchvision, scikit-learn, Matplotlib, Numpy, Pillow and tqdm.

> Commands below will install all dependencies:
>
> - pip install torch torchvision
> - pip install matplotlib
> - pip install pandas
> - pip install scikit-learn
> - pip install tqdm
>
> To install PyTorch with GPU support visit [this page](https://pytorch.org/get-started/locally/) for more details

---

### Model accuracy (validation set with noise) as of 27/06/26, 00:11:
>
> - ocr_v1_t0.045_v0.055.pt [411800]: *98.13%*
> - ocr_v5_t0.030_v0.025.pt [411800]: *98.72%*

---

### Model accuracy (wingdings) as of 26/06/26, 21:58:
>
> - ocr_wingding_t0.000_v0.000.pt: *100%*