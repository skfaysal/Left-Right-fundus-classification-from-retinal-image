<h2 align="center">Hi 👋, I'm Sheikh Md. Faysal</h2>
<h4 align="center">Machine Learning Engineer</h4>

<p align="left"> <img src="https://komarev.com/ghpvc/?username=skfaysal&label=Profile%20views&color=0e75b6&style=flat" alt="skfaysal" /> </p>

- 📫 How to reach me **skmdfaysal@gmail.com**

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://linkedin.com/in/md faysal" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="md faysal" height="20" width="30" /></a>
<a href="https://kaggle.com/faysal" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="faysal" height="20" width="30" /></a>
<a href="https://www.hackerrank.com/md faysal" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/hackerrank.svg" alt="md faysal" height="20" width="30" /></a>
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.docker.com/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://flask.palletsprojects.com/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/pocoo_flask/pocoo_flask-icon.svg" alt="flask" width="40" height="40"/> </a> <a href="https://git-scm.com/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

## Project Title

<h3 align="center">Left/Right fundus classification from retinal image</h3>

## Project Overview
Left/Right fundus classification is used to detect left and right fundus for making grader freandly user interface.It makes easier for grader to grade and screen our fundus images also it uses to filter good quality and usable fundus images. By using the training script we can train any deep learning model by doing some slight change in script. \
**N.B: All the models and weights here are kept dummy as it's sensitive and not shareable**.

## Run Locally

Clone the project

```bash
  git clone https://github.com/skfaysal/Left-Right-fundus-classification-from-retinal-image
```

Go to the project directory

```bash
  cd Right-fundus-classification-from-retinal-image.git
```

Create virtual environment using environement.yml

```bash
  conda env create -f environment.yml
```

Activate environment

```bash
  conda activate heat_map
```
For Training Model. We will pass parameters using CLI
```bash
  python3 lf_train_tfv1.py --img_row 224 --img_col 224 --epochs 25 --batch_size 16 --val_scplit 0.25
```
For Testing the model
```bash
  python3 test.py 
```
