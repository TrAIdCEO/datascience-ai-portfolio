# 🎭 Facial Emotion Recognition (FER)

**Harnessing Deep Learning to decode human sentiment through Artificial Emotional Intelligence (AEI).**

## **📌 Overview**

This project develops a high-accuracy Computer Vision pipeline to classify human facial expressions into four categories: **Happy, Neutral, Sad, and Surprise**. By identifying uniquely identifiable features—like the upward angle of a mouth or the width of the eyes—this model serves as a building block for applications in healthcare, market research, and human safety.

---

## **🛠️ Tech Stack**

| Category | Tools & Libraries |
| --- | --- |
| **Environment** | Jupyter Notebook, Conda (Local Runtime), Google Colab |
| **Deep Learning** | TensorFlow, Keras (Sequential API) |
| **Computer Vision** | OpenCV (logic), ImageDataGenerator (Augmentation) |
| **Data Science** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## **🔬 Model Benchmarking**

I experimented with custom CNN architectures and state-of-the-art Transfer Learning models. **Custom CNNs significantly outperformed pre-trained models** due to the specific grayscale nature of the dataset.

| Model | Accuracy | F1-Score | Key Takeaway |
| --- | --- | --- | --- |
| **CNN Model 3 (Final)** | **0.75** | **0.75** | **Top Performer.** Captured large spatial features effectively. |
| **CNN Model 2** | 0.70 | 0.71 | Strong, but struggled with Sad vs. Neutral classes. |
| **CNN Model 1** | 0.68 | 0.68 | Solid baseline for micro-expression detection. |
| **VGG16** | 0.52 | 0.52 | Sub-optimal; required RGB conversion which added noise. |
| **ResNet/EfficientNet** | N/A | N/A | Ineffective for this low-res, grayscale dataset. |

---

## **💡 Key Insights**

* **The Grayscale Advantage:** Choosing `color_mode = 'grayscale'` provided better performance than RGB. Since the raw images were grayscale, forcing them into 3 channels (for Transfer Learning) introduced superfluous data that hindered learning.
* **CNN vs. ANN:** CNNs were selected for their ability to automatically detect 2D spatial relationships between pixels (like eyebrow furrows) without human supervision.
* **Intuitive Architecture:** The final model succeeded by using a higher number of nodes in `Conv2D` and `Dense` layers, allowing it to capture complex spatial features across larger areas of the face.
* **Classification Challenges:** The model achieved near-perfect scores for **Happy** and **Surprise** (0.93 precision), but faced difficulty distinguishing between **Sad** and **Neutral** due to overlapping features and potential dataset labeling noise.

---

## **🚀 Future Scope**

* **Relabeling:** Manual review of Sad/Neutral images to reduce dataset noise.
* **Cross-Validation:** Implementing K-Fold validation to ensure model robustness.
* **Fine-Tuning:** Unfreezing specific layers in Transfer Learning models to adapt them more closely to FER.

---

## **📜 Author’s Note & Technical Context (June 2023)**
This project was developed and finalized in June 2023. Highlighting this date is essential for contextualizing the model's performance and the architectural choices made during the development phase.

### **🔬 The "75% Accuracy" Benchmark**
In the field of Computer Vision, an accuracy of 75% can sometimes appear modest. However, for the FER-2013 dataset (which this project utilizes), this score is highly competitive for the time:
* **Human Performance:** Human accuracy on this specific dataset—known for its "in-the-wild," low-resolution (48x48) images—is approximately 65% ± 5%. This model significantly outperforms the average human evaluator.
* **State-of-the-Art (2023):** At the time of creation, standard Deep CNN architectures typically plateaued between 71% and 75.2%. This project sits at the top of that performance bracket.
* **Dataset Difficulty:** The high degree of occlusion and the subtle differences between "Neutral" and "Sad" classes make this a notoriously difficult benchmark compared to more modern, high-definition datasets.

### **🛠️ Architectural Perspective**
The decision to prioritize Custom CNNs over Transfer Learning (like VGG16 or ResNet) was a strategic one based on the data available in 2023:
* **Data Specificity:** Pre-trained models are often optimized for RGB images. Since this dataset is natively grayscale, the custom-built cnn_model_3 was more efficient at extracting relevant facial landmarks without the noise introduced by channel conversion.
* **Technological Shift:** While Vision Transformers (ViT) and Self-Supervised Learning (SSL) became the dominant SOTA (State-of-the-Art) trends in late 2024 and 2025, this project serves as a robust demonstration of what can be achieved with optimized, "intuition-led" convolutional architectures.

---

## 🚀 How to Reproduce
*  **Environment Setup:**
    ```bash
    conda create -n fer_env python=3.10
    conda activate fer_env
    pip install tensorflow pandas matplotlib scikit-learn seaborn
    ```
*  **Data Loading:**
    * Ensure the `Facial_emotion_images.zip` is in your project directory and edit the file reference in the notebook to match it's location.
    * Run the notebook cells to extract and preprocess the image generators.
*  **Training:**
    * Open `Facial_Emotion_Detection_Full_Code.ipynb` in Jupyter or Google Colab.
    * Execute all cells, see **Building a Complex Neural Network Architecture** for training of the final model.

---
