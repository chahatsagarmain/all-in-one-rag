import os
import json
import time
import csv
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

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

from rag_builder.rag_builder import RAGBuilder
from chat.chat_manager import ChatManager
from chat.chat_client import GeminiChatClient
from evaluation.dataset_generator import generate_dataset

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_string(s: str) -> str:
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def evaluate_faithfulness(client: GeminiChatClient, context: str, answer: str) -> float:
    system_prompt = (
        "You are an objective judge evaluating a RAG pipeline's response for faithfulness (groundedness).\n"
        "Analyze the provided context and the answer. Decide if the answer contains any information "
        "not explicitly mentioned or directly logical from the context.\n"
        "Response format: Output ONLY a single numeric float score between 0.0 and 1.0.\n"
        "1.0 means the answer is 100% faithful and grounded. 0.0 means it is completely ungrounded or hallucinated.\n"
        "Do not include any other text or reasoning."
    )
    user_content = f"Context:\n{context}\n\nAnswer:\n{answer}"
    try:
        res_text = client.generate_response(system_prompt, [{"role": "user", "content": user_content}]).strip()
        match = re.search(r"\d?\.\d+", res_text)
        if match:
            return float(match.group())
        return float(res_text)
    except Exception as e:
        print(f"Warning: Faithfulness evaluation failed: {e}")
        return 0.5

def evaluate_answer_relevance(client: GeminiChatClient, query: str, answer: str) -> float:
    system_prompt = (
        "You are an objective judge evaluating a RAG pipeline's response for answer relevance.\n"
        "Analyze the user's query and the generated answer. Decide if the answer directly addresses the query.\n"
        "Response format: Output ONLY a single numeric float score between 0.0 and 1.0.\n"
        "1.0 means the answer is highly relevant and directly answers the query. 0.0 means it is completely irrelevant.\n"
        "Do not include any other text or reasoning."
    )
    user_content = f"Query:\n{query}\n\nAnswer:\n{answer}"
    try:
        res_text = client.generate_response(system_prompt, [{"role": "user", "content": user_content}]).strip()
        match = re.search(r"\d?\.\d+", res_text)
        if match:
            return float(match.group())
        return float(res_text)
    except Exception as e:
        print(f"Warning: Answer relevance evaluation failed: {e}")
        return 0.5

def calculate_retrieval_hit(retrieved_chunks: List[str], reference_context: str) -> float:
    ref = reference_context.lower().strip()
    if not ref:
        return 0.0
    for chunk in retrieved_chunks:
        chunk_clean = chunk.lower().strip()
        if ref in chunk_clean or chunk_clean in ref:
            return 1.0
        ref_words = set(ref.split())
        chunk_words = set(chunk_clean.split())
        if not ref_words:
            continue
        overlap = len(ref_words & chunk_words) / len(ref_words)
        if overlap >= 0.4:  # 40% word overlap counts as a match for recall
            return 1.0
    return 0.0

def run_evaluation(dataset_path: str, source_doc: str, num_queries: int = 5):
    print("=" * 80)
    print("🎯 STARTING MODULAR RAG PIPELINE EVALUATION")
    print("=" * 80)

    # 1. Load dataset (generate if not existing)
    if not os.path.exists(dataset_path):
        print(f"Dataset '{dataset_path}' not found. Generating synthetically...")
        generate_dataset(source_doc, dataset_path, count=num_queries)
        
    dataset = load_dataset(dataset_path)[:num_queries]
    print(f"Loaded {len(dataset)} queries for evaluation.")

    # 2. Check if Weaviate works
    weaviate_works = False
    try:
        from db.weaviate_store import WeaviateVectorStore
        store = WeaviateVectorStore()
        client = store.connect()
        # Ping Weaviate
        weaviate_works = client.is_ready()
        print("Weaviate Vector DB is running and connected.")
    except Exception as e:
        print(f"[Warning] Weaviate connection failed: {e}. Weaviate benchmarks will be skipped.")

    # 3. Define configuration matrix
    configurations = [
        {"name": "Local_Fixed_SimpleStore", "chunker": "fixed", "embedder": "local", "store": "simple"},
        {"name": "Local_Semantic_SimpleStore", "chunker": "semantic", "embedder": "local", "store": "simple"},
        {"name": "Static_Fixed_SimpleStore", "chunker": "fixed", "embedder": "static", "store": "simple"},
    ]
    if weaviate_works:
        configurations.append(
            {"name": "Local_Fixed_Weaviate", "chunker": "fixed", "embedder": "local", "store": "weaviate"}
        )

    eval_client = GeminiChatClient()
    results = []
    summary_stats = {}

    for config in configurations:
        print("\n" + "-" * 80)
        print(f"🚀 Benchmarking configuration: {config['name']}")
        print("-" * 80)

        # Build pipeline
        try:
            rag_pipeline = (
                RAGBuilder()
                .with_data_source(source_doc)
                .with_chunker(config["chunker"])
                .with_embedder(config["embedder"])
                .with_vector_store(config["store"])
                .build()
            )
            
            # Ingest, chunk, embed, load to store
            t0 = time.perf_counter()
            rag_pipeline.build()
            ingest_build_time = time.perf_counter() - t0
            print(f"Build phase completed in {ingest_build_time:.2f}s.")
        except Exception as e:
            print(f"[Error] Failed to build configuration {config['name']}: {e}")
            continue

        chat_manager = ChatManager(
            rag_pipeline=rag_pipeline,
            chat_client=eval_client
        )

        config_results = []

        for idx, item in enumerate(dataset):
            query = item["query"]
            ref_context = item["reference_context"]
            ground_truth = item["ground_truth"]

            print(f"[{config['name']}] Evaluating Query {idx + 1}/{len(dataset)}...")

            # Measure retrieval
            t_ret_start = time.perf_counter()
            retrieved_results = rag_pipeline.query(query, top_k=5)
            retrieval_latency = time.perf_counter() - t_ret_start

            ret_texts = [r["text"] if isinstance(r, dict) else str(r) for r in retrieved_results]
            
            # Measure overall generation
            t_gen_start = time.perf_counter()
            response = chat_manager.chat(query)
            generation_latency = time.perf_counter() - t_gen_start

            # Calculate metrics
            hit = calculate_retrieval_hit(ret_texts, ref_context)
            
            joined_context = "\n".join(ret_texts)
            faithfulness = evaluate_faithfulness(eval_client, joined_context, response)
            relevance = evaluate_answer_relevance(eval_client, query, response)

            row = {
                "Configuration": config["name"],
                "Query": query,
                "Ground Truth": ground_truth,
                "Retrieved Chunks": len(ret_texts),
                "LLM Response": response,
                "Build Time (s)": ingest_build_time,
                "Retrieval Latency (s)": retrieval_latency,
                "Generation Latency (s)": generation_latency,
                "Total Latency (s)": retrieval_latency + generation_latency,
                "Recall@5": hit,
                "Faithfulness": faithfulness,
                "Relevance": relevance
            }
            config_results.append(row)
            results.append(row)
            chat_manager.clear_history()

        # Compute summary stats
        avg_ret_lat = sum(r["Retrieval Latency (s)"] for r in config_results) / len(config_results)
        avg_gen_lat = sum(r["Generation Latency (s)"] for r in config_results) / len(config_results)
        avg_recall = sum(r["Recall@5"] for r in config_results) / len(config_results)
        avg_faithfulness = sum(r["Faithfulness"] for r in config_results) / len(config_results)
        avg_relevance = sum(r["Relevance"] for r in config_results) / len(config_results)

        summary_stats[config["name"]] = {
            "Avg Retrieval Latency (s)": avg_ret_lat,
            "Avg Generation Latency (s)": avg_gen_lat,
            "Recall@5": avg_recall,
            "Faithfulness": avg_faithfulness,
            "Relevance": avg_relevance
        }

    # 4. Save results to CSV
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / "evaluation_results.csv"
    try:
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    except PermissionError:
        csv_file = output_dir / f"evaluation_results_{int(time.time())}.csv"
        print(f"[Warning] Could not write to default CSV because it is locked/open. Saving to: {csv_file}")
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    summary_file = output_dir / "evaluation_summary.json"
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
    except PermissionError:
        summary_file = output_dir / f"evaluation_summary_{int(time.time())}.json"
        print(f"[Warning] Could not write to default JSON because it is locked/open. Saving to: {summary_file}")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)

    # 5. Print summary table
    print("\n" + "=" * 80)
    print("📈 EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<30} | {'Retr Lat':<8} | {'Gen Lat':<8} | {'Recall@5':<8} | {'Faithful':<8} | {'Relevance':<8}")
    print("-" * 80)
    for name, stats in summary_stats.items():
        print(f"{name:<30} | {stats['Avg Retrieval Latency (s)']:.4f}s | {stats['Avg Generation Latency (s)']:.4f}s | {stats['Recall@5']:.2f}     | {stats['Faithfulness']:.2f}     | {stats['Relevance']:.2f}")
    print("=" * 80)
    print(f"Detailed logs exported to: {csv_file}")
    print(f"Summary JSON exported to: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline configurations.")
    parser.add_argument("--dataset", type=str, default="./data/eval_dataset.json", help="Path to golden dataset")
    parser.add_argument("--source", type=str, default="./README.md", help="Path to reference doc to index")
    parser.add_argument("--count", type=int, default=5, help="Number of queries to run from dataset")
    
    args = parser.parse_args()
    run_evaluation(args.dataset, args.source, args.count)
