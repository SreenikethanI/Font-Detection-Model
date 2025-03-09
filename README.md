<h1 style="text-align: center; font-weight: bold; color: white; text-decoration: underline; font-variant: small-caps;">Machine Learning based Typeface Recognition Tool</h1>

A Deep Learning model to predict the font used in a given image.

Check out the repo for an interactive front-end demo:
<br />✨ https://github.com/joejo-joestar/Font-Detection-App

Check out the images in the [Demo sample](Demo%20sample/) folder to test the model.

Team members:
- **Sreenikethan Iyer - 2022A7PS0034U**
- **Joseph Cijo - 2022A7PS0019U**
- **Mohammed Emaan - 2022A7PS0036U**
- **Yusra Hakim - 2022A7PS0004U**

---

# Model architecture
<img src="Model%20layout.png" width="1000px"/>

---

# Environment setup for Jupyter notebook
Firstly install Miniconda from [here](https://docs.anaconda.com/miniconda/install/).

Then open a command prompt in this directory, and run the following. This will create and activate an environment called "FDS".

```bash
conda create -n fds python=3.12
conda activate fds
```

After running this, your CMD prompt should have a "`(fds)`" prefixed at the start.

Run the following command to install packages, such as [PyTorch](https://pytorch.org/get-started/locally/). This will take some time.

```bash
conda install -n fds ipykernel ipywidgets --update-deps --force-reinstall

pip install scikit-learn opencv-contrib-python-headless matplotlib numpy pandas pillow pyperclip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

You're ready! Now just open VS Code and open the [Jupyter notebook](Notebook.ipynb), and remember to select the FDS environment at the bottom-right.
