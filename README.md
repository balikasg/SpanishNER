# Code for NER labeling for meddoprof task

[Meddoprof](https://temu.bsc.es/meddoprof/) is a data science challenge concerning
Named Entity Recognition (NER) in Spanish medical texts.

This repo provides code that explores NER with Linear CRF. THe purposes are educational. 

# Data
```bash
curl -O https://zenodo.org/record/4775741/files/meddoprof-training-set.zip
unzip meddoprof-training-set.zip
```

This downloads and unzips the data in your current directory. Two new directories appear: 
`task1` and `task2` that contain the data for the two tasks. 

# Preprocessing
`feature_extraction.py` applies pre-processing to convert data in a training-friendly format. 
You need to first install the Python dependencies
```python
pip install -r requirements.txt
```

and then:
```python
python feature_extraction.py
```
