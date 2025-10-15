# 🧠 Emotion Classification using BERT — Full Fine-Tuning vs LoRA

This project compares **Full Fine-Tuning** and **LoRA (Low-Rank Adaptation)** of a BERT model on the **GoEmotions** dataset for multi-label emotion classification.  
It was developed as part of *Assignment 3: Fine-Tuning Pretrained Transformers for Emotion Classification* by **Jia Cheng Chung**.

---

## 📘 Project Overview
The project explores how different fine-tuning strategies affect model performance and training efficiency.

- **Dataset:** [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) — 58k Reddit comments labeled with 27 emotions + neutral  
- **Model:** `bert-base-uncased`
- **Tasks:**
  - Fine-tune BERT using **Full Fine-Tuning**
  - Fine-tune BERT using **LoRA**
  - Compare results (F1-score, accuracy, and loss trends)
  - Evaluate emotion predictions on custom text input

---

## 📄 Run Output
To make this project easy to review, a complete **run log (with all code cells executed and outputs displayed)** has been exported to PDF:

**Folder:** [run_output](run_output)

This folder includes:
- The full fine-tuning and LoRA training processes
- Loss curves, evaluation metrics, and plots
- Example predictions using `multilevel_pipeline`

You can open it directly on GitHub or download it for offline viewing.

---

## 🧩 Repository Structure
```
emotion-classification-finetuning/
│
├── full_finetuning.ipynb          ← Full fine-tuning notebook
├── lora_finetuning.ipynb          ← LoRA fine-tuning notebook
├── run_output.                    ← PDF showing all code, outputs, and results
    ├── full_finetuning.pdf
    ├── full_finetuning_10epochs.pdf
    ├── lora_finetuning.pdf
    └── lora_finetuning_10epochs.pdf
├── requirements.txt               ← Python dependencies
└── report_assg3.pdf               ← Final written report

```

---

## ⚙️ Setup Instructions

### 🖥️ Option 1 — Run Locally (Laptop / PC)
> 💡 Tested on **Python 3.10.11** with NVIDIA GeForce RTX 3050 Ti GPU.

1. **Clone this repository:**
   ```bash
   git clone https://github.com/JiachengCJC/Emotion-Classification-using-BERT---Full-Fine-Tuning-vs-LoRA.git
   cd Emotion-Classification-using-BERT---Full-Fine-Tuning-vs-LoRA
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open either notebook in **Jupyter Notebook**:
   - `full_finetuning.ipynb`
   - `lora_finetuning.ipynb`

4. Click **Run All** — this will:
   - Load the dataset  
   - Tokenize text  
   - Train and evaluate the model  
   - Display metrics and plots  

---

### ☁️ Option 2 — Run on Google Colab
If you prefer to use Google Colab (e.g., with a T4 GPU):

1. Upload either notebook (`full_finetuning.ipynb` or `lora_finetuning.ipynb`) to Colab.  
2. Go to **Runtime → Change runtime type → GPU**.  
3. Run the following cell **first** to install dependencies:
   ```python
   !pip install torch torchvision torchaudio accelerate datasets transformers scikit-learn matplotlib pandas numpy
   ```
4. Click **Runtime → Run all** to start training.

---

## 🧪 Testing Your Own Example
After training completes, you can test the model with your own text.

In the notebook:
```python
result = multilevel_pipeline("I hate you, but I also like you")  # Cell 17
print(predicted_emotions)  # Output appears in Cell 18
```

The model will output the predicted emotion labels and their confidence scores.

---

## 📊 Results Summary

| Method | Macro F1 | Accuracy | Notes |
|:--|--:|--:|:--|
| **Full Fine-Tuning** | 0.56 → 0.57 | 0.44 → 0.45 | Strong performance, faster convergence |
| **LoRA Fine-Tuning** | 0.12 → 0.43 | 0.06 → 0.29 | Lightweight, efficient on consumer GPUs |

---

## 💡 Key Insights
- Full fine-tuning achieves higher accuracy but requires more GPU memory.  
- LoRA performs worse initially but improves with a higher learning rate (5e-5) and more epochs (10).  
- Label imbalance in GoEmotions makes minority emotions (e.g., *grief*, *pride*) hard to learn.  
- LoRA is ideal for resource-limited setups (e.g., laptops, low-VRAM GPUs).

---

## 🧾 Citation
If you use this work, please cite:
> Chung, Jia Cheng. *Fine-Tuning Pretrained Transformers for Emotion Classification — A Comparison Between Full and LoRA Fine-Tuning*, 2025.

---

## 📬 Contact
For questions or collaboration:
**Jia Cheng Chung**  
📧 jiacheng.chung.work@gmail.com  
