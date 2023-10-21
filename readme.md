# Kidney and Liver Tumor Segmentation (KiTS-23 and LiTS-17 Dataset)
In this repository, we train an artifical neural network on slice-wise tumor segmentation, where each slice is extracted from CT volume scans.

## USER GUIDE
This repository aids to download the KiTS-23 and LiTS-17 datasets and train slice based segmentation neural network with custom parameters

### Setup Repository
To setup the repository on your local device, firstly:
```python
1. git clone https://github.com/ches-001/Kidney-and-Liver-Tumor-Segmentation

2. cd Kidney-Tumor-Segmentation-KiTS23-dataset-

3. pip install -r requirements.txt
```

### Download the KiTS-23 Dataset

Inside the `Kidney-and-Liver-Tumor-Segmentation` directory:
```python
1. git clone https://github.com/neheller/kits23

2. cd kits23

3. pip3 install --user -e .

4. cd ../

5. python download_dataset.py --dataset=kits23

6. python generate_metafile.py --dataset=kits23
```

### Download the LiTS-17 Dataset
To download the LiTS dataset, you must first create a kaggle account and generate a `kaggle.json` file on your [kaggle account settings](https://www.kaggle.com/settings). Copy and paste the downloaded `kaggle.json` file into the cloned local repository and follow the steps below:

```python
1. python download_dataset.py --dataset=lits17

2. python generate_metafile.py --dataset=lits17
```

**NOTE :** You can checkout the configuration arguments of the python files in step 8 and 9 via the `-h` flag like so:
```python
python download_dataset.py -h
```
To specify which dataset to download or generate its csv metafile, you can use the `--dataset` argument like so:
```python
python download_dataset.py --dataset=lits17

python generate_metafile.py --dataset=lits17
```

The unlike the LiTS17 dataset
### Usage

After setting up, you can then proceed to train the segmentation network like so:

```
python train.py
```

The `train.py` file has various CLI / configuration arguments, you can check them all out via the `-h` flag. To specify the dataset to train on, you can use the `--dataset` argument, similar to the download script.

**NOTE :** The trained model is stored in the `saved_model` directory in their respective subfolders, alongside a csv file corresponding to model performance during the course of training.
