# Kidney Tumor Segmentation (Kits23 Dataset)
In this repository, we train an artifical neural network on slice-wise tumor segmentation, where each slice is extracted from CT volume scans.

## USAGE
This repository aids to download the kits23 dataset and train slice based segmentation neural network with custom parameters

### Setup
To setup the repository on your local device, firstly:
```python
1. git clone https://github.com/ches-001/Kidney-Tumor-Segmentation-KiTS23-dataset-

2. cd Kidney-Tumor-Segmentation-KiTS23-dataset-

3. pip install -r requirements.txt

4. git clone https://github.com/neheller/kits23

5. cd kits23

6. pip3 install --user -e .

7. cd ../

8. python generate_metafile.py # (should be run if 'kits23_metafile.csv' is not available)

9. python download_dataset.py
```

**NOTE :** You can checkout the configuration arguments of the python files in step 8 and 9 via the `-h` flag like so:

```python
python download_dataset.py -h
```

### Usage

After setting up, you can then proceed to train the segmentation network like so:

```
python train.py
```

The `train.py` file has various CLI / configuration arguments, you can check them all out via the `-h` flag.

**NOTE :** The trained model is stored in the `saved_model` directory, alongside a csv file corresponding to model performance during the course of training.