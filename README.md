# Car Damage Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

This project detects and segments damaged parts in vehicle images.

## ✨ Key Features
- **High-Performance Model**: Achieves accurate segmentation by combining the DeepLabV3+ architecture with a powerful EfficientNet backbone.
- **Configuration-Driven Experiments**: Easily manage iterative experiments by changing hyperparameters and model options in a `train_config.yaml` file.
- **Folder-Based Inference**: Provides a convenient inference feature that processes not only single images but also entire folders at once.
- **Stable Logging**: Reliably records and manages the training process using `Loguru`.


## 📂 folder structure
The overall folder structure is as follows:

```
.
├── configs/
│   └── train_config.yaml
├── dataset/
│   └── dataset.py
├── experiments/
│   ├── run.py
│   └── predict.py
├── models/
│   ├── backbones/
│   ├── modules/
│   ├── __init__.py
│   └── deeplab.py
├── outputs/ (automatically generated)
│   └── (Stores experiment and prediction results)
├── trainer/
│   └── trainer.py
├── utils/
│   ├── helpers.py
│   ├── losses.py
│   └── metrics.py
├── predict.sh
└── requirements.txt
```

## 🔧 Setup and Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/clary97/Car_Damage_Segmentation.git
cd Car_Damage_Segmentation
```

#### 2. Download the Dataset

This project uses the **[Car Damage Dataset (CarDD)](https://cardd-ustc.github.io/)**. Please download the dataset from the link below and extract it so that the `CarDD_release` folder is located in the project's root/dataset directory.

* **Download Link**: https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view

The final folder structure should look like this:
```
Car_Damage_Segmentation/
├── dataset/
│   └── CarDD_release/
│       ├── CarDD_COCO
│       └── CarDD_SOD
├── configs/
├── experiments/
└── ...
```

#### 3. Create and Activate a Virtual Environment

It is highly recommended to create a virtual environment to keep project dependencies isolated.

```bash
conda create -n (your env name) python=3.8
conda activate (your env name)
```

#### 4. Install Dependencies

Once the virtual environment is activated, run the following command to install all the required libraries.

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training

All settings are managed in the `configs/train_config.yaml` file. To start an experiment, run the following command:

```bash
python experiments/run.py
```
As training progresses, a new timestamped folder will be created in the `outputs` directory, containing the trained model (`best_model.pth`), log files, and a final results summary (`summary_results.csv`).

### Inference

Open the `predict.sh` file, modify the paths for the image folder and model weights, and then run the following command:

```bash
sh predict.sh
```
Once execution is complete, the visualized result images with damage areas highlighted will be saved in the output folder specified in the `predict.sh` file.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).