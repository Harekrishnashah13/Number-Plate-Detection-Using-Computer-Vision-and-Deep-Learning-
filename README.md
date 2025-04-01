
# 🚘 Number Plate Detection using Computer Vision & Deep Learning

## 📌 Project Overview
This project focuses on detecting car number plates using computer vision and deep learning techniques. It aims to automate license plate recognition (LPR) by building an end-to-end pipeline from data processing to detection and visualization using convolutional neural networks (CNNs). This model can be used for real-time vehicle surveillance, smart traffic monitoring, and automated toll systems.

## 🧠 Technologies Used
- **Languages**: Python
- **Libraries**:
  - Data Handling: `pandas`, `numpy`, `glob`, `xml.etree.ElementTree`
  - Visualization: `matplotlib`, `seaborn`
  - Image Processing: `OpenCV`, `PIL`, `pytesseract`
  - Deep Learning: `TensorFlow`, `Keras`
  - Dataset Management: `datasets` from Hugging Face

---

## 🧾 Dataset
- **Source**: Hugging Face Datasets - `keremberke/license-plate-object-detection`
- **Format**: Annotated images with XML files in PASCAL VOC format
- **Images**: Contains labeled car images with bounding boxes around number plates

---

## 🔧 Project Structure
1. **Data Preparation**:
   - Downloaded the dataset using `datasets` library
   - Created structured folders for train, test, and validation sets
   - Parsed XML annotations to extract bounding box coordinates

2. **Preprocessing & Configuration**:
   - Image resizing to uniform dimensions
   - Normalization and augmentation for better model generalization
   - Created a configuration class for centralized control

3. **Model Building**:
   - Implemented a Convolutional Neural Network (CNN) using Keras
   - Used layers like Conv2D, MaxPooling, BatchNormalization, Dropout, and GlobalPooling

4. **Training**:
   - Optimized using `Adam` optimizer and `EarlyStopping`
   - Model checkpointing and tensorboard integration for monitoring

5. **Evaluation & Visualization**:
   - Measured accuracy, precision, recall, and F1-score
   - Displayed predictions with bounding boxes on test images

---

## 📈 Results
- The model was able to detect and localize number plates with high accuracy
- Implemented tesseract OCR to extract alphanumeric content from predicted bounding boxes (optional)

---

## 🚀 Key Highlights
- End-to-end deep learning pipeline for object detection
- Hands-on integration of Hugging Face datasets and TensorFlow models
- Practical exposure to real-world dataset annotation and parsing
- Transferable skills for applications in surveillance, compliance, and intelligent transport systems

---

## 📌 How to Run
1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`
3. Run the Jupyter notebook: `car-number-plate-detection-new.ipynb`

---

## 🔒 Future Enhancements
- Convert to YOLOv5 or YOLOv8 architecture for real-time performance
- Deploy as a web service using Flask/Streamlit
- Integrate OCR to extract license plate text with better accuracy

