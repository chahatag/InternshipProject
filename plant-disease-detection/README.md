# Leaf Disease Detection

## Objective
Classify tomato leaf images into 5 categories (including healthy) using a CNN model trained on the PlantVillage dataset.

## Tools Used
- Python
- TensorFlow / Keras
- OpenCV, NumPy, Matplotlib
- Streamlit (for web interface)

## Dataset
- Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Used Classes:
  - Tomato___Bacterial_spot
  - Tomato___Early_blight
  - Tomato___Late_blight
  - Tomato___Leaf_Mold
  - Tomato___healthy

## Workflow
1. Resize and normalize images
2. Train CNN on selected classes
3. Evaluate model using accuracy & F1-score
4. Build a Streamlit app to upload leaf images and show predictions
5. Save trained model as `plant_disease_model.h5`

## Run the App
```bash
streamlit run tomato_predict_app.py
