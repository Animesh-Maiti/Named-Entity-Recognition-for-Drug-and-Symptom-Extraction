
pip install torch transformers datasets seqeval scikit-learn pandas tqdm



import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, Features, Value, ClassLabel
import numpy as np
from seqeval.metrics import classification_report

bio_tags = [
 'O',
 'B-DRUG', 'I-DRUG',
 'B-SYMPTOM', 'I-SYMPTOM',
]

# Assuming you have your data in CoNLL format
def read_conll_file(file_path):
    sentences, labels = [], []
    current_sentence, current_labels = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                word, label = line.split('\t')
                current_sentence.append(word)
                current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []

    if current_sentence:  # Don't forget the last sentence
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels

train_sentences, train_labels = read_conll_file('train.tsv')
dev_sentences, dev_labels = read_conll_file('devel.tsv')
test_sentences, test_labels = read_conll_file('test.tsv')

def convert_to_dataset(sentences, labels, bio_tags):
    label_map = {tag: i for i, tag in enumerate(bio_tags)}

    # Convert string labels to IDs
    label_ids = [[label_map[label] for label in sentence_labels] for sentence_labels in labels]

    return Dataset.from_dict({
        "tokens": sentences,
        "ner_tags": label_ids
    }, features=Features({
        "tokens": [Value("string")],
        "ner_tags": [ClassLabel(names=bio_tags)]
    }))

train_dataset = convert_to_dataset(train_sentences, train_labels, bio_tags)
dev_dataset = convert_to_dataset(dev_sentences, dev_labels, bio_tags)
test_dataset = convert_to_dataset(test_sentences, test_labels, bio_tags)

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have word_id set to None
            if word_idx is None:
                label_ids.append(-100)  # -100 will be ignored in loss calculation
            # For the first token of a word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For subsequent subword tokens
            else:
                # If it's a B- tag, convert to I- for continuation
                if bio_tags[label[word_idx]].startswith("B-"):
                    i_tag = "I-" + bio_tags[label[word_idx]][2:]
                    label_ids.append(bio_tags.index(i_tag))
                else:
                    label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_dev = dev_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    "dmis-lab/biobert-v1.1",
    num_labels=len(bio_tags)
)

training_args = TrainingArguments(
    output_dir="./biobert-ner-custom",
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    warmup_steps=100,        # ← replaces warmup_ratio
    logging_steps=50,        # ← kept this, still valid
)

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [bio_tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [bio_tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = classification_report(true_labels, true_predictions, output_dict=True)

    # Extract metrics
    overall_f1 = results["micro avg"]["f1-score"]
    return {
        "precision": results["micro avg"]["precision"],
        "recall": results["micro avg"]["recall"],
        "f1": overall_f1,
    }

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

test_results = trainer.evaluate(tokenized_test)
print(f"Test results: {test_results}")

predictions, labels, _ = trainer.predict(tokenized_test)
predictions = np.argmax(predictions, axis=2)

# Remove ignored index 
true_predictions = [
    [bio_tags[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [bio_tags[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

print(classification_report(true_labels, true_predictions))

model.save_pretrained("./biobert-ner-final")
tokenizer.save_pretrained("./biobert-ner-final")

from google.colab import drive
drive.mount('/content/drive')

import shutil


shutil.copytree(
    "./biobert-ner-final",                          
    "/content/drive/MyDrive/biobert-ner-final"      
)

print("Model saved to Google Drive! ✅")

pip install PyPDF2

import gradio as gr
from PyPDF2 import PdfReader
import torch
from transformers import pipeline
import pandas as pd
import os
from google.colab import drive

# --- STEP 0: MOUNT DRIVE ---
# This ensures we can see the /content/drive folder
drive.mount('/content/drive', force_remount=True)

# --- STEP 1: SETUP THE PIPELINE ---
# Use absolute path to prevent Hugging Face Hub "Repo ID" errors
raw_path = "/content/drive/MyDrive/biobert-ner-final"
model_path = os.path.abspath(raw_path)

if not os.path.exists(model_path):
    print(f"❌ ERROR: Folder not found at {model_path}")
    print("Check if 'biobert-ner-final' is directly in your MyDrive folder.")
else:
    print(f"✅ Model found! Loading from: {model_path}")
    # Initialize the pipeline
    # device=0 uses the Colab T4 GPU, -1 uses CPU
    nlp = pipeline("ner",
                   model=model_path,
                   tokenizer=model_path,
                   aggregation_strategy="simple",
                   device=0 if torch.cuda.is_available() else -1)

# --- STEP 2: HELPER FUNCTIONS ---
def map_label(label_raw):
    label_map = {
        "LABEL_0": "O",
        "LABEL_1": "DRUG",
        "LABEL_2": "DRUG",
        "LABEL_3": "SYMPTOM",
        "LABEL_4": "SYMPTOM"
    }
    return label_map.get(label_raw, label_raw)

# --- STEP 3: MAIN PROCESSING LOGIC ---
def process_document(file):
    if file is None:
        return "No file uploaded", [], [], None

    # Step A: Extract Text from PDF
    try:
        reader = PdfReader(file.name)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    except Exception as e:
        return f"Error reading PDF: {str(e)}", [], [], None

    # Step B: Run Inference
    results = nlp(full_text)

    # Step C: Advanced Merging Logic (Fixes 'Am' + 'oxicillin')
    merged_results = []
    if results:
        current_ent = results[0].copy()
        current_ent['entity_group'] = map_label(current_ent['entity_group'])

        for next_ent in results[1:]:
            next_label = map_label(next_ent['entity_group'])

            # Merge if same label AND they are adjacent
            if next_label == current_ent['entity_group'] and next_ent['start'] <= current_ent['end'] + 2:
                current_ent['end'] = next_ent['end']
                current_ent['score'] = max(current_ent['score'], next_ent['score'])
            else:
                if current_ent['entity_group'] != "O":
                    merged_results.append(current_ent)
                current_ent = next_ent.copy()
                current_ent['entity_group'] = next_label

        if current_ent['entity_group'] != "O":
            merged_results.append(current_ent)

    # Step D: Format for UI
    highlights = []
    table_data = []
    last_idx = 0

    for entity in merged_results:
        if entity['start'] > last_idx:
            highlights.append((full_text[last_idx:entity['start']], None))

        entity_text = full_text[entity['start']:entity['end']].strip()
        label = entity['entity_group']

        highlights.append((entity_text, label))
        table_data.append([entity_text, label, f"{entity['score']:.2%}"])
        last_idx = entity['end']

    if last_idx < len(full_text):
        highlights.append((full_text[last_idx:], None))

    # Step E: Export CSV
    csv_path = "extracted_medical_entities.csv"
    if table_data:
        df = pd.DataFrame(table_data, columns=["Entity Name", "Category", "Confidence"])
        df.to_csv(csv_path, index=False)
    else:
        csv_path = None

    return full_text, highlights, table_data, csv_path

# --- STEP 4: GRADIO UI DESIGN ---
with gr.Blocks(theme=gr.themes.Soft(), title="BioBERT NER System") as demo:
    gr.Markdown("""
    # 🏥 BioBERT Medical Entity Extractor
    **B.Tech CSE (AI/ML) Mini-Project** | Automatic Drug & Symptom Extraction
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Medical Report (PDF)", file_types=[".pdf"])
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Extract Entities", variant="primary")

            gr.Markdown("### 📥 Export Results")
            download_file = gr.File(label="Download CSV Report")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("✨ Highlighted Analysis"):
                    output_highlight = gr.HighlightedText(
                        label="NER Visualizer",
                        combine_adjacent=True,
                        color_map={"DRUG": "#ffd700", "SYMPTOM": "#ff6b6b"}
                    )
                with gr.TabItem("📝 Raw Text Preview"):
                    output_text = gr.Textbox(label="Extracted Text", lines=15, interactive=False)

    gr.Markdown("### 📋 Entity Summary Table")
    output_table = gr.Dataframe(
        headers=["Entity Name", "Category", "Confidence"],
        datatype=["str", "str", "str"],
        label="Detected Entities"
    )

    submit_btn.click(
        fn=process_document,
        inputs=file_input,
        outputs=[output_text, output_highlight, output_table, download_file]
    )

    clear_btn.click(lambda: (None, "", [], [], None), None, [file_input, output_text, output_highlight, output_table, download_file])

# Launching with a public share link
demo.launch(debug=True, share=True)