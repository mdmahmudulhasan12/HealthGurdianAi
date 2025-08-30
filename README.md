# HealthGuardian AI: AI-Powered Brain Tumor & Skin Cancer Detection System

## Overview
HealthGuardian AI is an advanced medical diagnostic system leveraging **Artificial Intelligence (AI)** and **Deep Learning** to assist healthcare professionals in early and accurate detection of **brain tumors** and **skin cancer**. The project integrates two computer vision modules:

1. **Brain Tumor Detection**: Uses Convolutional Neural Networks (CNNs) to classify MRI scans and identify the presence and type of brain tumors.
2. **Skin Cancer Detection**: Employs a CNN-based model built with Keras to distinguish between malignant and benign skin lesions using dermoscopic images.

By combining these AI-driven systems, HealthGuardian AI aims to provide **reliable, fast, and accessible diagnostic support**, enhancing healthcare services globally.

## Technology Stack
- **Programming Languages**: Python  
- **Frameworks**: TensorFlow, Keras  
- **Libraries**: OpenCV, NumPy, Matplotlib, Scikit-learn, and more  
- **Deployment**: Flask on local device  
- **Models**: CNN, VGG16 for Brain Tumor Detection

# HealthGuardian AI - Project Structure

- **Main_project_interface.py** – Main Flask application connecting the trained models with the web interface.  
- **brain_tumor_model.h5** – Trained CNN model for brain tumor detection.  
- **skin_cancer_cnn.h5** – Trained CNN model for skin cancer detection.  
- **index.html** – Main webpage where users upload images and see results. Contains HTML and CSS code.  
- **skin-cancer-detection-with-cnn-deep-learning.ipynb** – Jupyter Notebook containing the full training process, data preprocessing, CNN model creation, and evaluation for skin cancer detection.  
- **brain_tumour_detection_using_deep_learning.ipynb** – Jupyter Notebook containing the complete workflow for brain tumor detection, including data preparation, CNN model training, and performance evaluation.  
- **README.md** – Documentation file explaining the project.


## Applications
- **Preliminary Diagnostic Tool**: Provides fast analysis of MRI and skin images to assist doctors.  
- **Rural Healthcare Support**: Enables screening in remote areas with limited specialists.  
- **Second Opinion System**: Reduces human error by cross-checking diagnoses.  
- **Telemedicine Integration**: Can be incorporated into telehealth platforms for remote patient screening.  
- **Medical Education and Training**: Serves as a learning tool for students and trainees.


## Installation
```bash
# Clone the repository
git clone https://github.com/mdmahmudulhasan12/HealthGurdianAi.git

# Navigate to the project directory
cd HealthGurdianAi
