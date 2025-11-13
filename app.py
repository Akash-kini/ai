import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import docx
import PyPDF2
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import threading
import pandas as pd

# --- Load AI text detection model (fallback if one fails) ---
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
except:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

def extract_text(file_path):
    """Extract text from TXT, DOCX, or PDF."""
    text = ""
    if file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return text.strip()

def check_plagiarism(text):
    """Improved plagiarism detection using chunk-based Google search."""
    chunks = [text[i:i+300] for i in range(0, len(text), 300)]  # divide text into 300-character chunks
    similarities = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for chunk in chunks:
        try:
            query = "+".join(chunk.split()[:20])  # first 20 words per chunk
            url = f"https://www.google.com/search?q={query}"
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            results = [r.get_text() for r in soup.find_all("span")]

            if not results:
                similarities.append(0)
                continue

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([chunk] + results)
            similarity = cosine_similarity(vectors[0:1], vectors[1:])
            similarities.append(similarity.max())
        except:
            similarities.append(0)

    if not similarities:
        return 0.0
    return round((sum(similarities) / len(similarities)) * 100, 2)

def check_ai_generated(text):
    """Detect AI-generated content using HuggingFace model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_score = scores[0][1].item() * 100
    return round(ai_score, 2)

def save_report(results):
    """Save results to Excel file."""
    df = pd.DataFrame(results, columns=["File Name", "Plagiarism (%)", "AI-generated (%)"])
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if save_path:
        df.to_excel(save_path, index=False)
        messagebox.showinfo("Saved", f"Report saved to:\n{save_path}")

def colorize_row(tree, item_id, plagiarism_score):
    """Color-code rows based on plagiarism level."""
    if plagiarism_score < 20:
        tree.tag_configure("low", background="lightgreen")
        tree.item(item_id, tags=("low",))
    elif 20 <= plagiarism_score < 50:
        tree.tag_configure("medium", background="yellow")
        tree.item(item_id, tags=("medium",))
    else:
        tree.tag_configure("high", background="tomato")
        tree.item(item_id, tags=("high",))

def process_files():
    """Handle file selection and analysis."""
    files = filedialog.askopenfilenames(
        title="Select Assignment Files",
        filetypes=(("Text files", "*.txt"), ("Word files", "*.docx"), ("PDF files", "*.pdf"))
    )
    if not files:
        return

    progress_bar["maximum"] = len(files)
    progress_bar["value"] = 0
    result_table.delete(*result_table.get_children())
    analysis_results.clear()

    def run_checks():
        for i, file in enumerate(files, start=1):
            text = extract_text(file)
            plagiarism_score = check_plagiarism(text)
            ai_score = check_ai_generated(text)

            row_id = result_table.insert("", "end", values=(
                os.path.basename(file),
                f"{plagiarism_score}%",
                f"{ai_score}%"
            ))
            colorize_row(result_table, row_id, plagiarism_score)
            analysis_results.append([os.path.basename(file), plagiarism_score, ai_score])

            progress_bar["value"] = i
            root.update_idletasks()

        messagebox.showinfo("Completed", "Plagiarism check completed for all files!")
        save_btn.config(state="normal")

    threading.Thread(target=run_checks).start()

# --- GUI Setup ---
root = tk.Tk()
root.title("AI Plagiarism Checker - Improved")
root.geometry("750x550")

label = tk.Label(root, text="Upload assignments to check plagiarism and AI content", wraplength=650, justify="center")
label.pack(pady=10)

upload_btn = tk.Button(root, text="Upload Files", command=process_files, width=20, height=2, bg="blue", fg="white")
upload_btn.pack(pady=10)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
progress_bar.pack(pady=10)

columns = ("File Name", "Plagiarism", "AI-generated")
result_table = ttk.Treeview(root, columns=columns, show="headings", height=10)
for col in columns:
    result_table.heading(col, text=col)
    result_table.column(col, width=220)
result_table.pack(pady=10)

analysis_results = []
save_btn = tk.Button(root, text="Save Report", command=lambda: save_report(analysis_results), state="disabled", bg="green", fg="white", width=20)
save_btn.pack(pady=10)

root.mainloop()
