# Named-Entity-Recognition-for-Drug-and-Symptom-Extraction


This project presents a Named Entity Recognition (NER) system for extracting drug names and symptoms from unstructured medical text. The system uses BioBERT, a transformer-based model pre-trained on biomedical data, and fine-tunes it for token classification using a labeled dataset with BIO tagging.


The data is preprocessed, tokenized, and aligned to handle subword tokens effectively. The model is trained using the HuggingFace Trainer API and evaluated using precision, recall, and F1-score, achieving an overall F1-score of approximately 0.87.


A user-friendly interface is developed using Gradio, allowing users to upload medical reports, visualize extracted entities, and download structured outputs. This project demonstrates the effectiveness of NLP techniques in automating medical information extraction.
