import os
import json
import argparse
import sys
from pathlib import Path

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeError
if sys.platform == 'win32':
    try:
        reconfigure_stdout = getattr(sys.stdout, 'reconfigure', None)
        if reconfigure_stdout is not None:
            reconfigure_stdout(encoding='utf-8')
        reconfigure_stderr = getattr(sys.stderr, 'reconfigure', None)
        if reconfigure_stderr is not None:
            reconfigure_stderr(encoding='utf-8')
    except Exception:
        pass

# Add the parent directory to sys.path so we can import the pipeline modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingester.ingester import DataSourceIngester
from chunker.chunker import FixedSizeChunker
from chat.chat_client import GeminiChatClient

def clean_json_string(s: str) -> str:
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def generate_dataset(file_path: str, output_path: str, count: int):
    print(f"Reading input file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # 1. Ingest text
    ingestor = DataSourceIngester()
    content = ingestor.ingest_data_from_path(file_path)
    
    # 2. Chunk text
    chunker = FixedSizeChunker(content, chunk_size=800, overlap=100)
    chunks_data = chunker.create_chunks()
    chunks = chunks_data["chunks"]
    
    print(f"Ingested file and split into {len(chunks)} chunks.")
    
    if not chunks:
        print("Error: No chunks extracted from document.")
        sys.exit(1)

    # Pick chunks spaced evenly to cover the entire document
    step = max(1, len(chunks) // count)
    selected_chunks = [chunks[i] for i in range(0, len(chunks), step)][:count]
    
    print(f"Generating {len(selected_chunks)} Q&A pairs using Gemini...")
    
    gemini_client = GeminiChatClient()
    
    dataset = []
    
    system_prompt = (
        "You are an AI assistant helping to build a golden evaluation dataset for a RAG pipeline. "
        "Your task is to generate a question-answer pair based ONLY on the provided text chunk. "
        "The question must be answerable using only the information in the chunk. "
        "Do not assume or extrapolate anything outside of the chunk."
    )
    
    for idx, chunk in enumerate(selected_chunks):
        print(f"Generating question {idx + 1}/{len(selected_chunks)}...")
        user_prompt = (
            f"Provided Text Chunk:\n"
            f"\"\"\"\n{chunk}\n\"\"\"\n\n"
            f"Based on the chunk above, generate a single query and its ground truth answer.\n"
            f"Format your response as a JSON object with the following keys:\n"
            f"- 'query': A specific question whose answer is fully contained in the chunk.\n"
            f"- 'reference_context': The exact sentence or short snippet of the chunk that directly answers the question.\n"
            f"- 'ground_truth': A detailed, accurate answer to the question based ONLY on the chunk.\n\n"
            f"Return ONLY the raw JSON object. Do not include any explanation or markdown formatting."
        )
        
        try:
            messages = [{"role": "user", "content": user_prompt}]
            response_text = gemini_client.generate_response(system_prompt, messages)
            
            clean_response = clean_json_string(response_text)
            item = json.loads(clean_response)
            
            # Ensure required keys exist
            if "query" in item and "ground_truth" in item:
                # If reference_context isn't filled properly, use a fallback
                if "reference_context" not in item or not item["reference_context"]:
                    item["reference_context"] = chunk[:200] + "..."
                dataset.append(item)
                print(f"Successfully generated question: {item['query'][:50]}...")
            else:
                print(f"Warning: Response missing required keys. Raw: {response_text}")
        except Exception as e:
            print(f"Error generating Q&A pair for chunk {idx}: {e}")
            
    # Save the dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated evaluation dataset with {len(dataset)} items at '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG golden dataset using Gemini client.")
    parser.add_argument("--file", type=str, default="./README.md", help="Path to input document")
    parser.add_argument("--output", type=str, default="./data/eval_dataset.json", help="Path to output JSON dataset")
    parser.add_argument("--count", type=int, default=10, help="Number of questions to generate")
    
    args = parser.parse_args()
    generate_dataset(args.file, args.output, args.count)
