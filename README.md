# E2AD

Official PyTorch Implementation of
"Anomaly Detection in Medical Images Using Encoder-Attention-2Decoders Reconstruction".

IEEE Transactions on Medical Imaging 2025. [paper](https://ieeexplore.ieee.org/document/10979458)

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.8.12
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA A100-PCIE (40GB) and NVIDIA Driver Version: 555.52.04. Same GPU and package version are recommended. 

## 2. Prepare Datasets
### OCT2017
Creat a new directory `../OCT2017`. Download ZhangLabData from [URL](https://data.mendeley.com/datasets/rscbjbr9sj/3).
Unzip the file, and move everything in `ZhangLabData/CellData/OCT` to `../OCT2017/`. The directory should be like:
```
|-- OCT2017
    |-- test
        |-- CNV
        |-- DME
        |-- DRUSEN
        |-- NORMAL
    |-- train
        |-- CNV
        |-- DME
        |-- DRUSEN
        |-- NORMAL
```

### APTOS
Creat a new directory `../APTOS`.
Download APTOS 2019 from [URL](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
Unzip the file to `../APTOS/original/`. Now, the directory would be like:
```
|-- APTOS
    |-- original
        |-- test_images
        |-- train_images
        |-- test.csv
        |-- train.csv
```
Run the following command to preprocess the data to `../APTOS/`.
```
python ./prepare_dataset/prepare_aptos.py --data-folder ../APTOS/original --save-folder ../APTOS
```
The directory would be like:
```
|-- APTOS
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.

### ISIC2018
Creat a new directory `../ISIC2018`.
Go to the ISIC 2018 official [website](https://challenge.isic-archive.com/data/#2018).
Download "Training Data","Training Ground Truth", "Validation Data", and "Validation Ground Truth" of Task 3.
Unzip them to `../ISIC2018/original/`. Now, the directory would be like:
```
|-- ISIC2018
    |-- original
        |-- ISIC2018_Task3_Training_GroundTruth
        |-- ISIC2018_Task3_Training_Input
        |-- ISIC2018_Task3_Validation_GroundTruth
        |-- ISIC2018_Task3_Validation_Input
```
Run the following command to preprocess the data to `../ISIC2018/`.
```
python ./prepare_dataset/prepare_isic2018.py --data-folder ../ISIC2018/original --save-folder ../ISIC2018
```
The directory would be like:
```
|-- ISIC2018
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.


### Br35H
Creat a new directory `../Br35H`.
Go to the kaggle [website](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).
Download "yes" and "no".
Unzip them to `../Br35H/original/`. Now, the directory would be like:
```
|-- Br35H
    |-- original
        |-- yes
        |-- no
```
Run the following command to preprocess the data to `../ISIC2018/`.
```
python ./prepare_dataset/prepare_br35h.py --data-folder ../Br35H/original --save-folder ../Br35H
```
The directory would be like:
```
|-- Br35H
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.

## 3. Run Experiments
Run experiments with default arguments.

APTOS
```
python e2ad_aptos.py --train_times 5 --gpu 0 --model_name E2AD --data_dir your/path/to/apotos/
```

OCT2017
```
python e2ad_oct.py  --train_times 5 --gpu 0 --model_name E2AD --data_dir your/path/to/oct/
```

Br35H
```
python e2ad_br35h.py --train_times 5 --gpu 0 --model_name E2AD --data_dir your/path/to/br35h/
```

ISIC2018
```
python e2ad_isic.py --train_times 5 --gpu 0 --model_name E2AD --data_dir your/path/to/i2ic/
```

### Acknowledgement
This repository primarily draws upon Repository [EDC](https://github.com/guojiajeremy/EDC). Here, we would like to extend our special thanks to [GuoJia](https://github.com/guojiajeremy) for releasing such convenient code, which has contributed to the community's research.
Guo's other works are also highly intriguing and open-sourced. We strongly recommend following their research and starring their repositories to support their contributions: [Recontrast-NIPS 2023](https://github.com/guojiajeremy/ReContrast) and [Dinomaly-CVPR 2025](https://github.com/guojiajeremy/Dinomaly)


### Recommendation
Recommended repo: https://github.com/M-3LAB/awesome-industrial-anomaly-detection, which collected awesome AD papers and is convient for me to follow new ideas in this field.

Also, [UniNet-CVPR2025](https://github.com/pangdatangtt/UniNet) has achieved 100% image-level AUC on APTOS, OCT2017 and ISIC2018 datasets in single-class AD setting. So, the future research direction in medical AD would be the research on small scale datasets, or few-shot and zero-shot settings.


