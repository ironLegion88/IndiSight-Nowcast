import os
import time
from pathlib import Path
from typing import Any
import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(module_name=__name__, log_sub_dir="agent")

# ---------------------------------------------------------
# Standalone worker function for Multiprocessing
# Must be outside the class to avoid Pickling errors
# ---------------------------------------------------------
def extract_markdown_worker(pdf_path: Path) -> tuple[str, str | None, str | None]:
    """CPU-bound task to extract Markdown from a PDF."""
    try:
        md_raw: Any = pymupdf4llm.to_markdown(str(pdf_path))
        if isinstance(md_raw, str):
            md_text = md_raw
        elif isinstance(md_raw, list):
            text_parts = []
            for block in md_raw:
                if isinstance(block, dict):
                    text_value = block.get("text")
                    if text_value:
                        text_parts.append(str(text_value))
                elif isinstance(block, str):
                    text_parts.append(block)
            md_text = "\n".join(text_parts)
        else:
            md_text = str(md_raw)
        return pdf_path.name, md_text, None
    except Exception as e:
        return pdf_path.name, None, str(e)


class RAGIngestionPipeline:
    def __init__(self, reset_collection: bool = True):
        self.docs_dir = Path("data/raw/policy_docs")
        self.collection_name = "indisight_policies"
        self.reset_collection = reset_collection
        
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.retry_attempts = int(os.getenv("RETRY_ATTEMPTS", "3"))
        self.retry_backoff_seconds = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0"))
        
        self.embedding_model_name = "BAAI/bge-large-en-v1.5"
        self.vector_size = 1024 
        
        logger.info(f"Loading {self.embedding_model_name} to RTX 4070...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=200)
        self._init_qdrant()

    @staticmethod
    def _require_env_vars(keys: list[str]) -> None:
        missing = [k for k in keys if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

    def _retry(self, operation_name: str, fn):
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed (attempt %s/%s): %s. Retrying in %.1fs",
                    operation_name,
                    attempt,
                    self.retry_attempts,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        raise RuntimeError(f"{operation_name} failed after {self.retry_attempts} attempts") from last_error

    def _init_qdrant(self):
        try:
            self._require_env_vars(["QDRANT_HOST", "QDRANT_PORT"])
            self.client = self._retry(
                "Qdrant client initialization",
                lambda: QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=20),
            )
            
            # Wipe existing collection if reset is True to prevent duplicates
            if self.reset_collection and self.client.collection_exists(self.collection_name):
                logger.warning(f"Resetting Qdrant collection: {self.collection_name}")
                self._retry(
                    f"Delete collection {self.collection_name}",
                    lambda: self.client.delete_collection(self.collection_name),
                )
            
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating new Qdrant collection: {self.collection_name}")
                self._retry(
                    f"Create collection {self.collection_name}",
                    lambda: self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                    ),
                )
                
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def run_pipeline(self):
        logger.info("--- Starting Multi-Core RAG Ingestion Pipeline ---")
        
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error("No PDFs found in data/raw/policy_docs.")
            return

        logger.info(f"Found {len(pdf_files)} PDFs. Launching CPU Process Pool...")
        
        # Leave 4 threads for the OS and GPU operations
        cpu_count = os.cpu_count() or 1
        max_cpu_workers = max(1, cpu_count - 4)
        
        all_docs_to_embed =[]
        start_time = time.time()

        # Step 1: Parallel CPU Extraction
        with ProcessPoolExecutor(max_workers=max_cpu_workers) as executor:
            futures = {executor.submit(extract_markdown_worker, pdf): pdf for pdf in pdf_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Markdown (CPU)"):
                name, md_text, error = future.result()
                
                if error:
                    logger.error(f"Failed extracting {name}: {error}")
                    continue
                    
                if md_text and len(md_text.strip()) > 0:
                    # Chunking is fast enough for the main thread
                    chunks = self.splitter.create_documents(
                        texts=[str(md_text)],
                        metadatas=[{"source": name, "type": "policy_doc"}]
                    )
                    all_docs_to_embed.extend(chunks)

        extraction_time = time.time() - start_time
        logger.info(f"Extracted {len(all_docs_to_embed)} chunks in {extraction_time:.2f} seconds.")

        # Step 2: Batched GPU Ingestion
        if all_docs_to_embed:
            logger.info("Firing batch embeddings to RTX 4070 and Qdrant...")
            batch_start = time.time()
            
            # Send larger batches to fully saturate the GPU VRAM
            batch_size = 200 
            for i in tqdm(range(0, len(all_docs_to_embed), batch_size), desc="Ingesting to Qdrant (GPU)"):
                batch = all_docs_to_embed[i:i + batch_size]
                self._retry(
                    f"Qdrant batch ingest [{i}:{i + len(batch)}]",
                    lambda b=batch: self.vector_store.add_documents(b),
                )
                
            ingest_time = time.time() - batch_start
            logger.info(f"GPU Ingestion complete in {ingest_time:.2f} seconds.")

        total_time = (time.time() - start_time) / 60
        logger.info(f"--- Pipeline Complete! Total time: {total_time:.2f} mins ---")

if __name__ == "__main__":
    # reset_collection=True wipes the interrupted run's data and starts completely fresh
    pipeline = RAGIngestionPipeline(reset_collection=True)
    pipeline.run_pipeline()