# HealthGuardian AI: AI-Powered Brain Tumor & Skin Cancer Detection System

## Overview
HealthGuardian AI is an advanced medical diagnostic system leveraging **Artificial Intelligence (AI)** and **Deep Learning** to assist healthcare professionals in early and accurate detection of **brain tumors** and **skin cancer**. The project integrates two computer vision modules:

1. **Brain Tumor Detection**: Uses Convolutional Neural Networks (CNNs) to classify MRI scans and identify the presence and type of brain tumors.
2. **Skin Cancer Detection**: Employs a CNN-based model built with Keras to distinguish between malignant and benign skin lesions using dermoscopic images.

By combining these AI-driven systems, HealthGuardian AI aims to provide **reliable, fast, and accessible diagnostic support**, enhancing healthcare services globally.

## Motivation
Early diagnosis of brain tumors and skin cancer significantly improves treatment outcomes.  

- **Brain Tumors**: Often fatal if not detected early; MRI analysis is time-consuming and requires skilled radiologists.  
- **Skin Cancer**: Early-stage symptoms are subtle and often misdiagnosed, especially in remote areas.  

HealthGuardian AI seeks to bridge the gap between **timely diagnosis** and **healthcare accessibility**, using AI to provide critical second opinions and support in underserved regions.

## Problem Statement
Medical professionals face increasing workloads and the challenge of analyzing complex medical images quickly. Many rural areas lack access to expert radiologists or dermatologists, delaying diagnosis and treatment.  

HealthGuardian AI addresses the following issues:

- **Brain Tumor Detection**: Manual MRI scan analysis is time-consuming and prone to error.  
- **Skin Cancer Detection**: Early-stage symptoms can be subtle, leading to late diagnoses.  

By automating detection with deep learning, the system improves accuracy and delivers **rapid, reliable results**.

### Complex Engineering Problem
| Attribute | Explanation |
|-----------|-------------|
| **Depth of knowledge required** | Expertise in computer vision, CNNs, deep learning frameworks (TensorFlow/Keras), and medical image preprocessing. |
| **Range of conflicting requirements** | Balancing high prediction accuracy, fast processing, patient data privacy, and model interpretability. |
| **Depth of analysis required** | Feature extraction, CNN fine-tuning, and evaluation using medical-grade datasets. |
| **Familiarity of issues** | Handling imbalanced medical datasets and implementing advanced augmentation techniques. |
| **Extent of applicable codes** | Compliance with ethical AI guidelines, HIPAA-like regulations, and international health standards. |
| **Stakeholder involvement** | Hospitals, labs, doctors, patients, and regulators with differing priorities and constraints. |
| **Interdependence** | Model performance relies on quality preprocessing, accurate CNN training, and reliable deployment. |

## Design Goals / Objectives
- Develop a CNN-based **Brain Tumor Detection** model using MRI datasets.  
- Build a CNN-based **Skin Cancer Detection** system with dermoscopic images.  
- Achieve high **accuracy, precision, and recall** for real-world medical applications.  
- Optimize **preprocessing and augmentation** for robust predictions.  
- Integrate both modules into a **single, user-friendly interface**.  
- Enable **fast on-device or cloud-based inference** for real-time healthcare usage.

## Applications
- **Preliminary Diagnostic Tool**: Provides fast analysis of MRI and skin images to assist doctors.  
- **Rural Healthcare Support**: Enables screening in remote areas with limited specialists.  
- **Second Opinion System**: Reduces human error by cross-checking diagnoses.  
- **Telemedicine Integration**: Can be incorporated into telehealth platforms for remote patient screening.  
- **Medical Education and Training**: Serves as a learning tool for students and trainees.

## Technology Stack
- **Programming Languages**: Python  
- **Frameworks**: TensorFlow, Keras  
- **Libraries**: OpenCV, NumPy, Matplotlib, Scikit-learn  
- **Deployment**: Cloud or on-device inference  

## Installation
```bash
# Clone the repository
git clone https://github.com/mdmahmudulhasan12/HealthGurdianAi.git

# Navigate to the project directory
cd HealthGurdianAi
