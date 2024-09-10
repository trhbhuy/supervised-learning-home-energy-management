# Supervised learning home energy management systems (SL - HEMS)

This repository contains the implementation for our paper: ["Real-time energy scheduling for home energy management systems with an energy storage system and electric vehicle based on a supervised-learning-based strategy"](https://doi.org/10.1016/j.enconman.2023.117340), published in the Energy Conversion and Management.

## Setup 

```bash
conda env create -n torchtf --file env.yml
conda activate torchtf
```


## Structure

```bash
.
├── data/                                  # Directory for data
│   ├── raw/                               # Raw input data
│   ├── processed/                         # Processed datasets
│   └── generated/                         # Generated datasets
└── src/
    ├── networks/                          # Contains network-related logic
    │   ├── __init__.py
    │   ├── dnn.py/
    │   └── resnetd.py/
    ├── solver/
    │   ├── methods/
    │   │   ├── data_loader.py
    │   │   ├── dataset_aggregation.py
    │   │   ├── feature_engineering.py
    │   │   ├── optimization.py
    │   │   └── util.py
    │   ├── platform/
    │   │   ├── components/                # Components of the system
    │   │   │   ├── __init__.py
    │   │   │   ├── electric_vehicle.py
    │   │   │   ├── energy_storage.py
    │   │   │   ├── renewables.py
    │   │   │   └── utility_grid.py
    │   │   ├── hems.py                    # HEMS optimization
    │   │   ├── test_env.py                # HEMS environment setup and management (for testing)
    │   │   └── util.py
    │   ├── utils/                         # General utility functions
    │   │   ├── __init__.py
    │   │   ├── file_util.py
    │   │   └── numeric_util.py
    │   ├── __init__.py
    │   └── config.py                      # Configuration file for parameters
    ├── utils/                             # High-level utility scripts
    │   ├── __init__.py
    │   ├── common_util.py
    │   ├── preprocessing_util.py
    │   ├── test_util.py
    │   └── train_util.py
    ├── data_generation.py                 # Data generation scripts
    ├── preprocessing.py                   # Data preprocessing scripts
    ├── test_model.py                      # Model testing scripts
    └── train_model.py                     # Model training scripts
```


## How to run

### 1. Data Generation
To generate the necessary data for training and testing:

```
python3 data_generation.py
```

### 2. Data Preprocessing
Prepare the data for training by running the preprocessing script:

```
python3 preprocessing.py
```

### 3. Training the ResnesD Model
Train the ResnesD model using the generated data:

```
python3 train_model.py --data_dir data/generated/ \\
                      --model dnn --batch_size 48 \\
                      --epochs 200 --learning_rate 0.005 --gpu_device 0 \\
                      --lr_decay_epochs 50 --use_early_stop --patience 20
```

### 4. Testing the Model
Test the trained model on the microgrid environment:

```
python3 test_model.py --env hems \\
                      --data_path data/processed/ObjVal.csv \\
                      --num_test_scenarios 26 \\
                      --pretrained_model dnn \\
                      --learning_rate 0.005 --batch_size 48 --epochs 200 \\
```

## Citation
If you find the code useful in your research, please consider citing our paper:
```
@article{Huy2023,
   author = {Truong Hoang Bao Huy and Huy Truong Dinh and Dieu Ngoc Vo and Daehee Kim},
   doi = {10.1016/j.enconman.2023.117340},
   issn = {01968904},
   journal = {Energy Conversion and Management},
   month = {9},
   pages = {117340},
   title = {Real-time energy scheduling for home energy management systems with an energy storage system and electric vehicle based on a supervised-learning-based strategy},
   volume = {292},
   year = {2023},
}
```
<!-- ## License
[MIT LICENSE](LICENSE) -->
