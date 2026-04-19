import os
import re
import time
import subprocess
import wikipediaapi
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict

# run: nohup .venv/bin/python run_synthetic_data.py > synthetic_data.log 2>&1 &

# --- Workshop Settings ---
DEMO = False  # Set to True for a quick 5-minute run
MODEL_ENGINE = "Qwen/Qwen2.5-7B-Instruct"
NUM_PAIRS_PER_FILE = 30
QUALITY_THRESHOLD = 7.0
HF_REPOSITORY_ID = "lmassaron/medical-cardiology-qa"

def init_dirs():
    """Scaffold the local data directory structure."""
    for d in ["medical_input", "generated", "curated", "final", "output"]:
        os.makedirs(f"data/{d}", exist_ok=True)

def clean_string(input_string):
    """Remove whitespace and bracketed citations."""
    cleaned_string = " ".join(input_string.split())
    cleaned_string = re.sub(r"\{.*?\}|\}", "", cleaned_string)
    return cleaned_string

def get_wikipedia_content(titles):
    """Crawl Wikipedia sections for relevant medical knowledge."""
    wiki_wiki = wikipediaapi.Wikipedia("MedicalBot/1.0", "en")
    extracted_texts = []
    print(f"Scraping content for {len(titles)} topics...")
    for title in tqdm(titles):
        page = wiki_wiki.page(title)
        if page.exists():
            # Add the summary
            extracted_texts.append(f"{page.title} : {clean_string(page.summary)}")
            # Add non-trivial sections
            for section in page.sections:
                if len(section.text) > 200:
                    extracted_texts.append(
                        f"{page.title} ({section.title}) : {clean_string(section.text)}"
                    )
    return extracted_texts

def start_vllm():
    """Start the vLLM OpenAI-compatible server in the background."""
    print(f"Starting vLLM server with model {MODEL_ENGINE}...")
    vllm_log = open("vllm_server.log", "w")
    proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_ENGINE,
            "--port",
            "8000",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.25",
            "--max-model-len",
            "8192",
            "--disable-log-stats",
            "--uvicorn-log-level",
            "error",
        ],
        stdout=vllm_log,
        stderr=vllm_log,
    )
    
    print("Waiting for vLLM server to be ready...")
    # Wait for the log to contain the ready message
    ready = False
    while not ready:
        if os.path.exists("vllm_server.log"):
            content = open("vllm_server.log").read()
            if "Starting vLLM server" in content or "Uvicorn running on" in content:
                ready = True
        if not ready:
            time.sleep(5)
    print("vLLM server ready!")
    return proc

def generate_config():
    """Generate the YAML config for the synthetic-data-kit."""
    yaml_content = f"""
# Master configuration file for Synthetic Data Kit

paths:
  input:
    txt: "data/medical_input"
  output:
    parsed: "data/output"
    generated: "data/generated"
    curated: "data/curated"
    final: "data/final"

vllm:
  api_base: "http://localhost:8000/v1"
  port: 8000
  model: "{MODEL_ENGINE}"
  max_retries: 3
  retry_delay: 1.0

ingest:
  default_format: "txt"
  youtube_captions: "auto"

generation:
  temperature: 0.7
  top_p: 0.95
  chunk_size: 1022
  overlap: 64
  max_tokens: 1024
  num_pairs: {NUM_PAIRS_PER_FILE}

cleanup:
  threshold: {QUALITY_THRESHOLD}
  batch_size: 4
  temperature: 0.3

format:
  default: "jsonl"
  include_metadata: true
  pretty_json: true

prompts:
  summary: |
    As an expert cardiologist, summarize this document in 7-10 sentences, focusing on the main topic, key concepts,
    key diagnostic criteria, relevant pathophysiology, and first-line treatment where applicable.
  qa_generation: |
    You are an expert cardiologist writing training data for a medical AI.
    Based on the following medical text, create {{num_pairs}} high-quality question-answer pairs.
    Each pair should focus on diagnostic criteria, pathophysiology, or treatment guidelines.

    Rules:
    1. Questions should be clear, concise, and clinically relevant.
    2. Answers should be detailed (3-5 sentences), accurate, and based on the provided text.
    3. Include key diagnostic criteria and first-line treatment where applicable.
    4. Output in JSON format as a list of objects: [{{{{ "question": "...", "answer": "..." }}}}]
    5. Provide ONLY the JSON list, no preamble or explanation.

    Text: {{text}}
  qa_rating: |
    As an expert cardiologist, rate each of these question-answer pairs on a scale from 1-10, based on accuracy, clinical completeness, and relevance.

    YOU MUST RETURN A VALID JSON ARRAY OF OBJECTS WITH THIS EXACT SCHEMA:
    [
      {{{{ "question": "Exact question text", "answer": "Exact answer text", "rating": 8 }}}},
      ...
    ]

    *** YOUR RESPONSE MUST BE VALID JSON AND NOTHING ELSE - NO EXPLANATION, NO MARKDOWN ***

    QA pairs to rate:
    {{pairs}}
"""
    with open("medical_data_config.yaml", "w") as f:
        f.write(yaml_content)

def run_pipeline():
    init_dirs()
    
    # 1. Scrape
    cardiology_topics = [
        "Hypertension", "Acute coronary syndrome", "Myocardial infarction", 
        "Stable angina", "Unstable angina", "Coronary artery disease", 
        "Coronary artery spasm", "TIMI risk score", "Percutaneous coronary intervention",
        "Coronary artery bypass surgery", "Heart failure",
        "Heart failure with preserved ejection fraction", "Dilated cardiomyopathy",
        "Hypertrophic cardiomyopathy", "Restrictive cardiomyopathy",
        "Takotsubo cardiomyopathy", "Cardiac resynchronization therapy",
        "Atrial fibrillation", "Atrial flutter", "Supraventricular tachycardia",
        "Ventricular tachycardia", "Ventricular fibrillation",
        "Wolff-Parkinson-White syndrome", "Long QT syndrome", "Brugada syndrome",
        "Sick sinus syndrome", "Atrioventricular block", "Cardiac pacemaker",
        "Implantable cardioverter-defibrillator", "Aortic stenosis",
        "Aortic regurgitation", "Mitral stenosis", "Mitral regurgitation",
        "Mitral valve prolapse", "Tricuspid regurgitation", "Infective endocarditis",
        "Transcatheter aortic valve replacement",
        "Pericarditis", "Cardiac tamponade", "Beck's triad (cardiac tamponade)",
        "Myocarditis", "Constrictive pericarditis", "Aortic aneurysm",
        "Aortic dissection", "Peripheral artery disease",
        "Deep vein thrombosis", "Pulmonary embolism", "Dyslipidemia",
        "Familial hypercholesterolemia", "Atherosclerosis", "Metabolic syndrome",
        "Congenital heart disease", "Atrial septal defect", "Ventricular septal defect", 
        "Tetralogy of Fallot", "Patent ductus arteriosus", "Pulmonary hypertension",
        "Cor pulmonale", "Electrocardiography", "Echocardiography", "Cardiac stress test",
        "Cardiac catheterization", "Holter monitor", "Troponin", 
        "B-type natriuretic peptide", "Beta blocker", "ACE inhibitor", 
        "Calcium channel blocker", "Statin", "Anticoagulant", "Antiarrhythmic agent",
        "Diuretic", "Digoxin", "Nitrate (pharmacology)",
    ]
    
    if DEMO:
        cardiology_topics = cardiology_topics[:3]
        
    extracted_texts = get_wikipedia_content(cardiology_topics)
    for i, text in enumerate(extracted_texts):
        with open(f"data/medical_input/med_{i}.txt", "w") as f:
            f.write(text)

    # 2. Inference & Curation
    generate_config()
    vllm_proc = start_vllm()
    
    try:
        input_files = sorted([
            os.path.join("data/medical_input", f)
            for f in os.listdir("data/medical_input")
            if f.endswith(".txt")
        ])
        
        print(f"Processing {len(input_files)} files through synthetic-data-kit...")
        for filepath in tqdm(input_files, desc="Synthetic Generation"):
            basename = os.path.basename(filepath).replace(".txt", "")
            gen_file = os.path.join("data/generated", f"{basename}_qa_pairs.json")
            curated_file = os.path.join("data/curated", f"{basename}_qa_pairs_cleaned.json")
            final_file = os.path.join("data/final", f"{basename}_qa_pairs_cleaned_ft.json")

            # Step 1: Create raw pairs
            if not os.path.exists(gen_file):
                subprocess.run(
                    ["synthetic-data-kit", "-c", "medical_data_config.yaml", "create", filepath, "--type", "qa"],
                    check=True
                )

            # Step 2: Curate (Judge) pairs
            if not os.path.exists(curated_file) and os.path.exists(gen_file):
                subprocess.run(
                    ["synthetic-data-kit", "-c", "medical_data_config.yaml", "curate", gen_file],
                    check=True
                )

            # Step 3: Save in Fine-Tuning (FT) format
            if not os.path.exists(final_file) and os.path.exists(curated_file):
                subprocess.run(
                    ["synthetic-data-kit", "-c", "medical_data_config.yaml", "save-as", curated_file, "-f", "ft"],
                    check=True
                )
    finally:
        print("Terminating vLLM server...")
        vllm_proc.terminate()
        vllm_proc.wait()

    # 3. Final Formatting & HF Push
    print("Merging files and formatting dataset...")
    SYSTEM_PROMPT = "You are a knowledgeable medical assistant specializing in cardiology. Answer clinical questions accurately, focusing on diagnostic criteria, treatment guidelines, and pathophysiology."

    def set_system_prompt(row):
        if "messages" in row and len(row["messages"]) > 0:
            row["messages"][0]["content"] = SYSTEM_PROMPT
        return row

    final_files = [n for n in os.listdir("data/final") if n.endswith("_ft.json")]
    if not final_files:
        print("No processed files found in data/final!")
        return

    all_dfs = []
    for n in final_files:
        all_dfs.append(pd.read_json(f"data/final/{n}"))
    
    conversations = pd.concat(all_dfs).reset_index(drop=True)
    conversations = conversations.apply(set_system_prompt, axis=1)

    complete_dataset = DatasetDict({"train": Dataset.from_pandas(conversations)})
    complete_dataset.save_to_disk("data/medical-cardiology-qa")
    print(f"Final dataset ready: {len(conversations)} clinical pairs.")

    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"Pushing dataset to Hugging Face Hub: {HF_REPOSITORY_ID}...")
        complete_dataset.push_to_hub(HF_REPOSITORY_ID, token=token)
        print("Push complete!")
    else:
        print("HF_TOKEN not found in environment. Skipping Hub push.")

if __name__ == "__main__":
    run_pipeline()
