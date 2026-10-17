# Fruit Ripeness Detection

Classifies apple, banana and orange as ripe or unripe using classical image
processing: HSV/Otsu segmentation, colour, texture (GLCM, LBP) and shape
features, and a two-stage reference-vector classifier.

## Run
pip install -r requirements.txt
streamlit run app.py

## Dataset layout
dataset/dataset1/train/{apples,banana,orange}/{raw,ripe}/*.jpg
(optional) dataset/dataset1/test/...