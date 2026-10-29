# Pretrained Networks and Transfer Learning

Networks trained on millions of photos can be downloaded and reused. In these notebooks you classify photos with pretrained networks, then reuse their convolutional layers for a new task: a new dense head on their features, data augmentation and optional fine-tuning. An exercise applies the workflow to photos of sea animals, and an optional notebook tunes the hyperparameters with Keras Tuner.

## Learning Objectives

By the end of this repository, you should be able to:

- Classify a photo with a pretrained network from `keras.applications`.
- Extract features with a frozen pretrained base and train a dense head on them.
- Train a frozen pretrained base with data augmentation layers.
- Compare the models on the validation set and evaluate the chosen one on the test set.

## Learning Path

Work through the notebooks in this order:

| File / Folder | Description |
|---|---|
| [**01 - Pretrained Networks**](01_pretrained_networks.ipynb) | Inspect VGG16 and MobileNetV2 and classify photos with MobileNetV2. |
| [**02 - Transfer Learning**](02_transfer_learning.ipynb) | Classify bean leaf diseases with a dense head on pretrained features, augmentation and optional fine-tuning. |
| [**03 - Exercise: Classify Sea Animals with Transfer Learning**](03_exercise_sea_animals.ipynb) | Train a dense head and a model with augmentation on five sea animals from Caltech-101, and test the chosen one. |
| [**04 - Hyperparameter Tuning with Keras Tuner (Optional)**](04_optional_keras_tuner.ipynb) | Search the head of a transfer-learning model with Keras Tuner. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Solutions**](solutions/) | Worked solutions for all four notebooks. |
| [**Data**](data/) | The bean leaf and sea animal photos, with their sources and licences. |
| [**Assets**](assets/) | The figures, the scripts that draw them and the example photos. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
> **placeholder**. Replace it, including the `< >` brackets, with your own
> value. For example, `cd <repo-name>` becomes `cd amle-pretrained-networks-transfer-learning`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies, including TensorFlow and Keras, and creates a virtual environment in `.venv/`. TensorFlow is a large package, so the first install takes a few minutes.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## Data and Credits

The image data is part of the repository, in `data/` (23 MB). The pretrained weights are downloaded the first time a notebook uses them and stored in `~/.keras/models`: about 70 MB in total, loaded from disk on later runs.

- **Bean disease dataset** (`data/beans`, resized to 160 × 160 pixels, and one photo in notebook 01): Makerere AI Lab, [AI-Lab-Makerere/ibean](https://github.com/AI-Lab-Makerere/ibean), MIT licence.
- **Caltech-101** (`data/sea_animals` and the example photos in `assets/photos`): Fei-Fei, L., Fergus, R. and Perona, P., [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02), CC BY 4.0.

## References & Further Reading

- [**Keras Applications**](https://keras.io/api/applications/): Every pretrained network in Keras, with its size and accuracy.
- [**Keras guide: Transfer learning and fine-tuning**](https://keras.io/guides/transfer_learning/): The complete workflow, including how to handle batch normalisation.
- [**TensorFlow tutorial: Data augmentation**](https://www.tensorflow.org/tutorials/images/data_augmentation): Augmentation with preprocessing layers.
- [**Keras Tuner: Getting started**](https://keras.io/keras_tuner/getting_started/): Search spaces, tuners and results.
- [**Deep Learning with Python**](https://deeplearningwithpython.io/): François Chollet's book on Keras, free to read online; chapter 8 covers pretrained networks.
- [**Very Deep Convolutional Networks for Large-Scale Image Recognition**](https://arxiv.org/abs/1409.1556): The VGG paper.
- [**MobileNetV2: Inverted Residuals and Linear Bottlenecks**](https://arxiv.org/abs/1801.04381): The MobileNetV2 paper.
