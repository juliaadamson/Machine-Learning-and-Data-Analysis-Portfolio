# Machine Learning & Data Analysis Portfolio

I created these four projects as part of my university Artificial Intelligence module to explore how the choice of model, preprocessing and evaluation techniques changes depending on the type of data. Each project focuses on a different data modality: tabular, image, audio and text.

I developed the projects in Python using Jupyter Notebook and compared classical machine learning with deep learning approaches across the four datasets. I performed exploratory data analysis, applied data-specific preprocessing and feature extraction techniques, evaluated model performance using appropriate metrics, and explored model behaviour using explainability techniques such as feature importance, SHAP and Grad-CAM.

The projects use publicly available datasets from the UCI Machine Learning Repository and Kaggle and were developed using libraries and frameworks including scikit-learn, TensorFlow, PyTorch and Hugging Face Transformers.

> Each notebook contains the full workflow from preprocessing and model training through evaluation, visualisation and interpretation.

## Dataset & Results Summary

| Project | Data Type | Dataset Size | Classes | Task | Best Performing Model | Accuracy |
|---|---|---:|---:|---|---|---|
| Salary Predictor | Tabular | 48,842 records | 2 | Binary classification | Gradient Boosting | 86.99% |
| Leaf Analyser | Image | 20,638 images | 15 | Multi-class classification | ResNet50 | 95.32% |
| Sound Analyser | Audio | 8,732 clips | 10 | Multi-class classification | SVM (RBF) | 92.78% |
| Real vs Fake News | Text | 44,898 articles | 2 | Binary classification | DistilBERT | 99.89% |

## Projects

### 1. Salary Predictor

**Dataset:** [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) - UCI Machine Learning Repository  
**Task:** Binary classification

This project predicts whether an individual's income exceeds $50K per year using demographic and employment data. I performed exploratory data analysis to understand patterns within the dataset and preprocessed the mixture of categorical and numerical features through cleaning, encoding and scaling. I then compared four classical machine learning models with an artificial neural network to investigate how different approaches performed on structured tabular data, and used permutation feature importance to explore which factors contributed most strongly to the predictions.

**Techniques explored:**
- Data cleaning and missing-value handling
- Exploratory Data Analysis (EDA)
- Categorical encoding and feature scaling
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Artificial Neural Network (ANN)
- Accuracy, precision, recall, F1 and ROC-AUC evaluation
- Permutation feature importance
- K-Means and Gaussian Mixture Model clustering
- PCA visualisation

**Results:**
- Best classical model: **Gradient Boosting**
- Accuracy: **86.99%**
- ROC-AUC: **0.924**
- ANN ROC-AUC: **0.907**
- Unsupervised clustering showed weak alignment with the true income groups

[View notebook](Task%201%20Salary%20Predictor.ipynb)

---

### 2. Leaf Analyser

**Dataset:** [Plant Disease Recognition](https://www.kaggle.com/datasets/emmarex/plantdisease) - Kaggle  
**Task:** Multi-class image classification

This project classifies plant leaf images by species and disease type using traditional machine learning and deep learning approaches. I extracted colour and texture features to represent visual characteristics associated with plant disease, then compared these feature-based models with a ResNet50-based model. ResNet50 was used because its pretrained ImageNet weights allowed the model to transfer previously learned visual features to the leaf classification task. Grad-CAM visualisations were then used to investigate which regions of the leaves contributed most strongly to the model’s predictions.

**Techniques explored:**
- Image resizing and normalisation
- Data augmentation
- Colour histogram feature extraction
- Gabor texture features
- Random Forest
- Support Vector Machine (SVM)
- k-Nearest Neighbours (k-NN)
- Transfer learning with ResNet50
- Accuracy, precision, recall, F1 and ROC-AUC evaluation
- Learning curves and confusion matrices
- Grad-CAM explainability

**Results:**
- Best classical model: **Random Forest**
- Random Forest accuracy: **86.40%**
- Random Forest ROC-AUC: **0.9898**
- ResNet50 accuracy: **95.32%**
- ResNet50 F1-score: **0.9446**
- ResNet50 ROC-AUC: **0.9983**

[View notebook](Task%202%20Leaf%20Analyser.ipynb)

---

### 3. Sound Analyser

**Dataset:** [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) - Kaggle  
**Task:** Multi-class audio classification

This project classifies urban environmental sounds using both traditional machine learning and deep learning approaches. I extracted 32 audio features, including MFCCs, chroma and spectral contrast, to capture different characteristics of each sound such as spectral shape, pitch and frequency contrast. I then compared several classical machine learning models with a convolutional neural network (CNN), and used SHAP and feature importance analysis to investigate which audio features contributed most to the predictions.

**Techniques explored:**
- MFCC feature extraction
- Chroma features
- Spectral contrast
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- CNN-based classification
- SHAP explainability
- Feature importance analysis
- K-Means clustering
- PCA visualisation

**Results:**
- Best classical model: **SVM (RBF)**
- SVM test accuracy: **92.78%**
- SVM F1-score: **0.9253**
- CNN test accuracy: **91.71%**
- CNN F1-score: **0.9150**
- Jackhammer and siren were among the easiest classes to identify
- Car horn and street music were among the most difficult
- K-Means clustering showed poor alignment with the true sound classes

[View notebook](Task%203%20Sound%20Analyser.ipynb)

---

### 4. Real vs Fake News

**Dataset:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) - Kaggle  
**Task:** Binary text classification

This project investigates fake-news detection using traditional natural language processing and transformer-based deep learning. I used TF-IDF to represent articles based on the importance of words within the text and compared several classical machine learning classifiers using these features. I then fine-tuned DistilBERT, which can capture the context and relationships between words, to compare traditional feature-based methods with a transformer-based approach.

**Techniques explored:**
- Exploratory text analysis
- Text cleaning and stemming
- Stop-word removal
- TF-IDF vectorisation
- Logistic Regression
- Naïve Bayes
- Linear SVM
- DistilBERT fine-tuning
- Precision, recall, F1 and ROC-AUC evaluation
- Word-level feature importance
- K-Means clustering

**Results:**
- Best classical model: **Linear SVM**
- Linear SVM accuracy: **99.52%**
- Linear SVM F1-score: **0.9950**
- DistilBERT accuracy: **99.89%**
- DistilBERT F1-score: **0.9988**
- DistilBERT ROC-AUC: **0.9999**
- Unsupervised clustering showed poor separation between real and fake articles

[View notebook](Task%204%20Real%20vs%20Fake%20News.ipynb)

## Technologies

| Area | Tools & Techniques |
|---|---|
| Programming | Python |
| Data Analysis | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras, PyTorch |
| NLP | Hugging Face Transformers, TF-IDF |
| Computer Vision | ResNet50, CNNs |
| Audio Processing | MFCCs, chroma, spectral contrast |
| Visualisation | Matplotlib, Seaborn |
| Explainability | Permutation Importance, SHAP, Grad-CAM |
| Development | Jupyter Notebook |

## Key Skills Demonstrated

- Exploratory data analysis and data preprocessing
- Feature engineering and feature extraction
- Classical machine learning
- Deep learning and transfer learning
- Computer vision
- Audio classification
- Natural language processing
- Model evaluation and comparison
- Explainable AI
- Unsupervised learning and clustering
- Data visualisation
