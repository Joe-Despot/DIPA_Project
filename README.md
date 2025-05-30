# DIPA_Project

## Project directory structure:
dataset/
── train/
   ── images/
      ── G1...jpg
      ── G2...jpg
   ── annotations.csv
── test/
   ── images/
      ── G1...jpg
      ── G2...jpg
   ── annotations.csv
── valid/
   ── images/
      ── G1...jpg
      ── G2...jpg
   ── annotations.csv
project_custom_cnn.ipynb
project_faster_rcnn.ipynb
project_yolo_v8.ipynb
pothole_data.yaml
README.md
requirements.txt

## Project description

The project is an analysis of 3 seperate models for image recognition. The 3 models used are YOLOv8, FasterRCNN and a custom CNN model.
Each of the 3 models and their training are separted into 3 distinct jupyter notebook files.
Each file is organized in 3 sections: 
1.  Model Initialization
  - lmodel data class, value conversion
2. Model Training
    - yolo using internal implementation
    - fasterrcnn and custom cnn
        - backpropagation and gradient descent
3. Model evaluatinon
    - visualization of 10 examples of ground truth and predicted image
    - metrics like accuracy, recall, f1, precision

The output of each jupyter notebook is a .pdf file with side by side examples of ground truth and predicted boundingboxes and metrics for each model (accuracy, recall, f1, precision).
The dataset used was: [https://universe.roboflow.com/hackthethong/pothole-detection-gmnid/], YOLOv8 OBB format and [] for the _annotations.csv file

