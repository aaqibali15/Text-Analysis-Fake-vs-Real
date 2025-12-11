# 📰 Task 4: Fake News Detection System

## 📌 Project Overview
This project focuses on **Natural Language Processing (NLP)** to detect fake news articles. It compares **Classic Machine Learning** models (Logistic Regression) against **Deep Learning** techniques (Fine-Tuning **DistilBERT**) to classify news as either "Real" or "Fake" with high accuracy.

The project also explores **Unsupervised Learning (LDA)** for topic modelling and includes **Explainability** analysis to understand the linguistic patterns driving the model's decisions.

## 📊 Dataset
The project uses the **Fake and Real News Dataset** combined with a custom dataset hosted on Google Drive.
* **Source:** Kaggle & Custom Data
* **Data Link:** [Download Dataset via Google Drive](https://drive.google.com/drive/folders/1v4nsvAsPekhySMsuh02FSonm_YWNpedI?usp=sharing)
* **Contents:**
    * `Fake.csv`: Articles flagged as unreliable or fake.
    * `True.csv`: Verified real news articles.
## 🚀 Key Features
* **Data Preprocessing:** Robust cleaning pipeline (removing URLs, punctuation, stop words).
* **Classic ML Baseline:** Logistic Regression and Naïve Bayes models using **TF-IDF**.
* **Deep Learning (SOTA):** Fine-tuned **DistilBERT (Transformer)** model using Hugging Face.
* **Unsupervised Learning:** Latent Dirichlet Allocation (LDA) to discover hidden topics in news.
* **Explainability:** Feature importance analysis visualizing words that trigger "Fake" vs "Real" classification.
* **Inference Pipeline:** A ready-to-use function to predict credibility of any new headline.

## 📊 Dataset
The project uses the **Fake and Real News Dataset** from Kaggle, containing approximately 45,000 articles.
* `Fake.csv`: Articles flagged as unreliable or fake.
* `True.csv`: Verified real news articles.

## 🛠️ Installation & Requirements

To run this project, you need **Python 3.x**. 

**Note on Versions:** This project requires specific library versions (specifically `numpy<2.0`) to avoid conflicts with `pandas` and `transformers`.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fake-news-detection.git](https://github.com/your-username/fake-news-detection.git)
    cd fake-news-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *If you don't have a requirements file, use:*
    ```bash
    pip install "numpy<2.0" pandas matplotlib seaborn scikit-learn torch "transformers" "accelerate>=0.26.0" nltk
    ```

## 🖥️ Usage

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Task4_FakeNews.ipynb
    ```
2.  **Run All Cells:** The notebook is structured sequentially:
    * **Step 1:** Data Loading & Cleaning.
    * **Step 2:** Training Classic ML Models.
    * **Step 3:** Fine-Tuning DistilBERT (Deep Learning).
    * **Step 4:** Unsupervised Topic Modelling (LDA).
    * **Step 5:** Visualization & Inference.

## 📈 Model Performance / Results

We compared two primary approaches. Deep Learning provided state-of-the-art performance, while Classic ML offered a highly efficient baseline.

| Model | Accuracy | F1-Score | Training Time |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 99.0% | 0.99 | < 1 min |
| **DistilBERT (Transformer)** | **99.75%** | **0.99** | ~20 mins (CPU) |

**Key Insight:** Feature analysis revealed that fake news is often characterized by sensationalist words like *breaking, video, wow,* and *hillary*, whereas real news uses formal attributions like *reuters, washington,* and *said*.

## 🤝 Contributors
* **[Your Name]** - *Classic ML, Deep Learning & Analysis*
* **[Colleague Name]** - *[Mention their contribution, e.g., Model Optimization / Research]*

## 📄 License
This project is for educational purposes as part of the Artificial Intelligence Coursework (MOD004553).
