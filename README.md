# 📚 Book Genre Classification System (Multimodal AI using RoBERTa-Large)

## 🔍 Overview
This project is a **multimodal AI system** that predicts the **primary genre of a book** using its **text description, cover image, or audio narration**. Built using **state-of-the-art NLP and deep learning models**, the system integrates **OCR for images**, **speech-to-text for audio**, and **transformer-based classification** for text.

The core NLP model is **RoBERTa-Large**, fine-tuned on a **cleaned and balanced Goodreads dataset**, achieving competitive results on a challenging **multiclass classification** problem.

## ✨ Key Features
* **Text Classification:** Fine-tuned **RoBERTa-Large** for genre prediction from book titles & descriptions.
* **OCR Integration:** Extracts text from book cover images using **Pytesseract**.
* **Speech-to-Text:** Converts audio narrations to text using **SpeechRecognition** and **PyDub**.
* **Balanced Dataset Handling:** Removed noisy samples and applied TF-IDF outlier filtering.
* **Deployment Ready:** Model is saved in HuggingFace format and integrated into a **Streamlit dashboard**.
* **Colab-Friendly:** Entire workflow runs on **Google Colab** with GPU acceleration.

## 📊 Dataset
* **Source:** Goodreads dataset containing book titles, descriptions, and genres.
* **Preprocessing Steps:**
  * Removal of rare genres (<50 samples).
  * Balanced sampling (up to 2000 books per genre).
  * Text cleaning, punctuation normalization, and whitespace removal.
  * TF-IDF based outlier removal.
* **Final Classes:** ~15 genres.

## 🛠 Tech Stack
* **Languages:** Python
* **Libraries & Frameworks:**
  * NLP: HuggingFace Transformers, Datasets, PyTorch
  * ML: Scikit-learn, NumPy, Pandas
  * Audio: SpeechRecognition, PyDub
  * Image: Pillow, Pytesseract
  * Visualization: Matplotlib, Seaborn
  * Deployment: Streamlit
* **Environment:** Google Colab

## ⚙️ Model Training
1. **Model Architecture:** RoBERTa-Large fine-tuned for sequence classification.
2. **Tokenization:** Max length = 256 tokens, padding & truncation applied.
3. **Training Setup:**
   * Optimizer: AdamW
   * Learning Rate: 2e-5
   * Batch Size: 16
   * Epochs: 8
   * Early stopping patience: 2 epochs
   * Mixed precision (FP16) when GPU available.
4. **Evaluation Metrics:** Accuracy & Weighted F1-score.

## 📈 Performance & Results

### Model Performance Metrics
| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 67.76% | **69.11%** |
| F1 Score | 0.6703 | **0.6875** |

**Note:** This is a **highly challenging 15-class classification problem** with many overlapping genres. The model maintains **balanced performance** across classes, indicating strong generalization.

### System Demonstrations
The FEEL2READ application successfully demonstrates multimodal genre prediction across different input types:

<div align="center">

| Text Input | Voice Input |
|:----------:|:-----------:|
| ![Text-based Genre Prediction](https://github.com/user-attachments/assets/9f9a6835-43c0-4b50-9f61-823d08157084) | ![Voice-based Genre Prediction](https://github.com/user-attachments/assets/2a98eefc-6310-4fce-9818-107586361ce1) |
| **Fig 1:** Genre prediction from book descriptions | **Fig 2:** Voice-to-text genre classification |

| Image Input | Mood Input |
|:-----------:|:----------:|
| ![Image-based Genre Prediction](https://github.com/user-attachments/assets/d53dc514-14e2-4488-b30a-976ef265e9fe) | ![Mood-based Genre Prediction](https://github.com/user-attachments/assets/71ff2b7d-106d-445e-b013-c25aeaafc5b5) |
| **Fig 3:** OCR-based text extraction and classification | **Fig 4:** Mood-based genre recommendations |

</div>

#### Key Features Demonstrated:
- **Text Processing:** Direct input of book titles and descriptions with confidence scoring
- **Voice Integration:** Real-time speech-to-text conversion supporting multiple audio formats  
- **Image Analysis:** OCR technology extracting text from book covers and synopsis images
- **Mood Mapping:** Innovative feature connecting emotional states to appropriate genres
- **Multi-Genre Support:** Successfully classifies across genres including Comedy, Romance, Thriller, Mystery, Fantasy, Drama, and Paranormal

## 📂 Project Structure
```
📦 BookGenreClassification
 ┣ 📜 major_project.ipynb      # Main Colab notebook
 ┣ 📜 README.md                # Project documentation
 ┣ 📂 dataset/                 # Goodreads dataset (optional, link if large)
 ┣ 📂 best_model/              # Fine-tuned RoBERTa-Large model
 ┣ 📂 images/                  # Sample cover images for testing
 ┗ 📂 dashboard/               # Streamlit app files
```

## 🚀 How to Run

### On Google Colab
1. Upload the notebook and dataset to your Colab environment.
2. Install required dependencies:
```bash
!pip install transformers datasets streamlit pytesseract pillow SpeechRecognition pydub scikit-learn
```
3. Run the notebook cells to train or load the model.
4. Launch the Streamlit dashboard:
```bash
!streamlit run dashboard.py
```

### Local Setup
1. Clone this repository:
```bash
git clone https://github.com/yourusername/BookGenreClassification.git
cd BookGenreClassification
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run dashboard/app.py
```

## 🎯 Applications
* Automated genre tagging for digital libraries.
* Enhanced recommendation systems.
* Audiobook & eBook classification.
* Content curation for publishing platforms.

## 🔧 Requirements
```
transformers>=4.21.0
datasets>=2.4.0
torch>=1.12.0
scikit-learn>=1.1.0
streamlit>=1.12.0
pytesseract>=0.3.9
Pillow>=9.2.0
SpeechRecognition>=3.8.1
pydub>=0.25.1
pandas>=1.4.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
```
## 🙏 Acknowledgments
* HuggingFace for the transformers library
* Goodreads for the dataset
* The open-source community for the various libraries used
