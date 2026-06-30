# Software AT3
## Due Term 2 Week 11, Wednesday 1st July 2026

Goal is to create a good CNN to perform optical character recognition. ***Running on Python 3.12.9 in a virtual environment***. Execution does not require a virtual environment however it is suggested.

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

### How to run

In order to run the program you can execute main.py. This program should not be executed in IDLE as it will break the progress bars.

If you want to run a prediction using the wingdings model you will have to remove the labels when graphing results within `Main.show_test_results()`. This is because the program does not fully support it.

---

### Training

The training dataset for ocr_v1-5 consists of Bree Serif, EGB Garamond, Georgia, Palatino Linotype, Merriweather, Times New Roman, Arial, Calibri, Comfortaa, Monsterrat, Oxygen, Verdana, Consolas, Courier New, Google Sans Code, Roboto Mono, Source Code Pro, Aclonica, Bowlby One SC, Comic Sans MS, Permanent Marker, Saira Stencil, Caveat, Creepster, Fontdiner Swanky, Homemade Apple, Pacifico, Yellowtail, Average, Maname, Rubik Glitch, Salsa, Sassy Frass, Aguafina Script, Festive.

All models were designed for level 4 complexity with noise, rotation, bold, and italics.

---

### Model accuracy as of 27/06/26:
>
> - ocr_v1_t0.045_v0.055.pt: *97.57%*
> - ocr_v2_t0.026_v0.032.pt: *98.25%*
> - ocr_v3_t0.027_v0.030.pt: *98.45%*
> - ocr_v4_t0.051_v0.030.pt: *98.35%*
> - ocr_v5_t0.008_v0.004.pt: *99.82%*
> - ocr_wingding_t0.000_v0.000.pt: *100.00%*
