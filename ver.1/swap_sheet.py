import os
import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, FilterSelector, Filter, FieldCondition, MatchValue 
from qdrant_client.http.exceptions import UnexpectedResponse # คงบรรทัดนี้ไว้
import uuid
import grpc
# --- Google Sheet Imports ---
import gspread
from google.oauth2.service_account import Credentials
import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

import requests

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# --- Environment Variable Loading ---
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), 'key.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info("Loaded environment variables from key.env")
    else:
        load_dotenv()
        logging.info("Loaded environment variables from .env or system environment")
except ImportError:
    logging.warning("python-dotenv not installed. Environment variables will not be loaded from files.")
except Exception as e:
    logging.error(f"Error loading environment variables: {e}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found. Gemini processing will not work.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Gemini API configured successfully.")


# --- Jina AI Configuration (ปรับปรุงข้อความ log ให้ชัดเจนขึ้น) ---
JINA_API_KEY = os.getenv("JINA_API_KEY") # อาจจะยังคงไว้เพื่อความเข้ากันได้ย้อนหลัง หรือลบออกได้
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-clip-v2") # ไม่ได้ใช้แล้วแต่คงไว้

if not JINA_API_KEY:
    logging.info("JINA_API_KEY not found. Jina AI embedding generation is not used in this setup (using local Ollama/Qwen).")
else:
    logging.info("Jina AI API key found but not used for embeddings in this setup (using local Ollama/Qwen).")


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# เปลี่ยนชื่อ Collection หลักเป็น PREFIX แทน
QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "documents_") 
# QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents") # อันนี้จะไม่ได้ใช้แล้ว

# เพิ่ม Environment Variable สำหรับ Ollama/Qwen Embedding Service URL
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", "http://192.168.1.10:11434/api/embeddings")


qdrant_client = None
if QDRANT_URL:
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            # check_compatibility=False, # อาจเพิ่มตรงนี้เพื่อปิด warning ถ้ายังเจอ
        )
        logging.info("Qdrant client initialized for cloud instance.")
        # Removed collection creation here as it will be done per-sheet dynamically
    except Exception as e:
        logging.error(f"Error initializing Qdrant client with cloud URL: {e}", exc_info=True)
        exit(1)
else:
    logging.error("QDRANT_URL not found. Qdrant client will not be initialized for cloud.")
    exit(1)

GOOGLE_SHEET_KEY_PATH = os.getenv("GOOGLE_SHEET_KEY_PATH")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_SHEET_WORKSHEET_NAMES = [name.strip() for name in os.getenv("GOOGLE_SHEET_WORKSHEET_NAME", "").split(",") if name.strip()]
GOOGLE_SHEET_UNIQUE_ID_COLUMN = os.getenv("GOOGLE_SHEET_UNIQUE_ID_COLUMN")
GOOGLE_SHEET_LAST_UPDATED_COLUMN = os.getenv("GOOGLE_SHEET_LAST_UPDATED_COLUMN")

gc = None
worksheets = [] # อันนี้ไม่ได้ถูกใช้งานตรงๆ แล้ว

if GOOGLE_SHEET_KEY_PATH and GOOGLE_SHEET_ID and GOOGLE_SHEET_WORKSHEET_NAMES:
    try:
        creds = Credentials.from_service_account_file(
            GOOGLE_SHEET_KEY_PATH,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        gc = gspread.authorize(creds)
        # sh = gc.open_by_key(GOOGLE_SHEET_ID) # ไม่ต้องเปิดที่นี่แล้ว จะเปิดตอนเรียกใช้ใน get_google_sheet_data
        logging.info("Google Sheet client authorized successfully.")

        # ไม่ต้องโหลด worksheet ตรงนี้แล้ว จะโหลดตอนเรียกใช้ใน get_google_sheet_data
    except Exception as e:
        logging.error(f"Error initializing Google Sheet client or loading worksheets: {e}")
else:
    logging.warning("Google Sheet configuration is incomplete. Sheets will not be loaded.")


# --- Utility Functions ---
async def async_process_text_with_gemini(
    input_text: str,
    row_num: int,
    model: genai.GenerativeModel,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, str | None, Dict[str, int] | None]:
    if model is None:
        logging.error(
            f"async_process_text_with_gemini (Row {row_num}): Gemini model is None. Skipping text processing.")
        return row_num, None, None

    async with semaphore:
        logging.info(
            f"Starting Gemini processing for row {row_num} (text-only)...")
        markdown_content = None
        usage_metadata: Dict[str, int] | None = None

        try:
            logging.info(
                f"           Calling Gemini API for row {row_num}...")
            # Prompt ที่เหมาะสำหรับการปรับแต่งข้อความหรือสรุปข้อมูล
            prompt_text = (
                "Please refine and structure the following text content as Markdown. "
                "Ensure that all information is accurately represented, and any inherent structure "
                "(like lists, tables, or sections) is converted into appropriate Markdown syntax. "
                "Do not add any introductory or concluding remarks, just the refined content. "
                "If the text is empty or meaningless, return an empty string."
                f"\n\nText Content:\n{input_text}"
            )
            try:
                await asyncio.sleep(0.3)  # Add a delay before calling Gemini API
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt_text # ส่งแค่ข้อความ
                )
                markdown_content = response.text
                if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
                    usage_metadata = {
                        'prompt_token_count': response.usage_metadata.prompt_token_count,
                        'candidates_token_count': response.usage_metadata.candidates_token_count,
                        'total_token_count': response.usage_metadata.total_token_count
                    }
                    prompt_tokens = usage_metadata.get('prompt_token_count', 0)
                    candidates_tokens = usage_metadata.get('candidates_token_count', 0)
                    total_tokens = usage_metadata.get('total_token_count', 0)

                    logging.info(
                        f"            Row {row_num} Token Usage: Prompt={prompt_tokens}, Output={candidates_tokens}, Total={total_tokens}")
                else:
                    logging.warning(
                        f"            Row {row_num}: No usage_metadata found in API response.")
                    usage_metadata = None
                logging.info(
                    f"Row {row_num} processing complete ({len(markdown_content) if markdown_content else 0} characters)")
            except Exception as gemini_e:
                logging.error(
                    f"async_process_text_with_gemini (Row {row_num}): Error calling Gemini API: {gemini_e}")
                if hasattr(gemini_e, 'response') and gemini_e.response:
                    if hasattr(gemini_e.response, 'usage_metadata') and gemini_e.response.usage_metadata is not None:
                        try:
                            usage_metadata = {
                                'prompt_token_count': gemini_e.response.usage_metadata.prompt_token_count,
                                'candidates_token_count': gemini_e.response.usage_metadata.candidates_token_count,
                                'total_token_count': gemini_e.response.usage_metadata.total_token_count
                            }
                            prompt_tokens = usage_metadata.get('prompt_token_count', 0)
                            candidates_tokens = usage_metadata.get('candidates_token_count', 0)
                            total_tokens = usage_metadata.get('total_token_count', 0)

                            logging.warning(
                                f"            Row {row_num} Token Usage before error: Prompt={prompt_tokens}, Output={candidates_tokens}, Total={total_tokens}")
                        except Exception as inner_e:
                            logging.warning(
                                f"            Row {row_num}: Could not retrieve usage_metadata from response error: {inner_e}")
                            usage_metadata = None
                    else:
                        logging.error(
                            f"Gemini API response error details: {gemini_e.response}")
                        usage_metadata = None
                if hasattr(gemini_e, 'response') and hasattr(gemini_e.response, 'prompt_feedback') and gemini_e.response.prompt_feedback:
                    logging.error(
                        f"            Row {row_num} Prompt Feedback: {gemini_e.response.prompt_feedback}")
                return row_num, None, usage_metadata
        except Exception as error:
            logging.error(
                f"async_process_text_with_gemini (Row {row_num}): Unexpected error during text processing: {error}", exc_info=True)
            markdown_content = None
            usage_metadata = None
        finally:
            pass

        return row_num, markdown_content, usage_metadata

# Removed chunk_text as it's no longer needed for row-based embedding

async def async_generate_embedding(
    input_data: List[str] # Changed to List[str] as we will pass full text
) -> Optional[List[List[float]]]:
    """
    Generates embeddings using a local embedding API (e.g., Qwen/Ollama) for a list of text inputs.
    Each item in input_data should be a string representing the text to embed.
    Returns a list of embeddings, where each embedding is a List[float].
    """
    if not input_data:
        logging.warning("Cannot generate embedding for empty input data.")
        return None

    url = OLLAMA_EMBEDDING_URL # ใช้ Environment Variable ที่กำหนดไว้
    headers = {
        "Content-Type": "application/json"
    }

    embeddings = []
    for text_to_embed in input_data: # Iterate over the list of strings
        payload = {
            "model": "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest",
            "prompt": text_to_embed # Use the string directly
        }
        try:
            response = await asyncio.to_thread(
                requests.post,
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            resp_json = response.json()
            if "embedding" in resp_json and isinstance(resp_json["embedding"], list):
                embeddings.append(resp_json["embedding"])
            else:
                logging.warning(f"No embedding returned for input: {text_to_embed}")
                embeddings.append(None)
        except Exception as e:
            logging.error(f"Error getting embedding for input: {text_to_embed} | {e}")
            embeddings.append(None)

    if embeddings and all(e is not None for e in embeddings):
        logging.info(f"Generated {len(embeddings)} embeddings successfully.")
        return embeddings
    else:
        logging.warning("Some embeddings failed to generate.")
        return embeddings

# --- Qdrant Save Function ---
async def save_points_to_qdrant(
    points: List[PointStruct],
    collection_name: str
):
    """
    Saves a list of PointStructs to Qdrant in a single upsert operation.
    """
    if qdrant_client is None:
        logging.warning("Qdrant client is not configured. Cannot save data.")
        return

    if not points:
        logging.info("No points to save to Qdrant.")
        return

    try:
        await asyncio.to_thread(
            lambda: qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True # รอให้ operations เสร็จสิ้น
            )
        )
        logging.info(f"✅ Qdrant upserted {len(points)} points to collection '{collection_name}' successfully.")
    except Exception as e:
        logging.error(
            f"❌ Error saving batch to Qdrant collection '{collection_name}': {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during Qdrant batch save to '{collection_name}': {e}"
        )

# --- Google Sheet Processing Functions ---
async def get_google_sheet_data(sheet_name: str) -> List[Dict[str, Any]]:
    """
    Connects to a specific Google Sheet worksheet and retrieves all data
    as a list of dictionaries. Each dictionary represents a row with column headers as keys.
    """
    if gc is None:
        logging.error("Google Sheet client is not initialized. Cannot retrieve data.")
        return []

    if not GOOGLE_SHEET_ID:
        logging.error("Google Sheet ID is not configured. Cannot retrieve data.")
        return []

    try:
        spreadsheet = await asyncio.to_thread(gc.open_by_key, GOOGLE_SHEET_ID)
        try:
            worksheet = await asyncio.to_thread(spreadsheet.worksheet, sheet_name)
            data = await asyncio.to_thread(worksheet.get_all_records, head=2)
            logging.info(f"Retrieved {len(data)} rows from worksheet '{sheet_name}'.")
            
            # Add sheet name to each row for context, it will be used by process_google_sheet_row_for_embedding
            for row in data:
                row["__worksheet__"] = sheet_name

            return data
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Worksheet '{sheet_name}' not found in spreadsheet '{GOOGLE_SHEET_ID}'. Skipping.")
            return []
        except Exception as sheet_error:
            logging.error(f"Error reading worksheet '{sheet_name}': {sheet_error}", exc_info=True)
            return []

    except gspread.exceptions.SpreadsheetNotFound:
        logging.error(f"Google Sheet with ID '{GOOGLE_SHEET_ID}' not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Google Sheet with ID '{GOOGLE_SHEET_ID}' not found.")
    except Exception as e:
        logging.error(f"Error fetching data from Google Sheet: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching data from Google Sheet: {e}")

async def process_google_sheet_row_for_embedding(
    row_data: Dict[str, Any],
    row_index: int,
    sheet_name: str,
    qdrant_target_collection: str # เพิ่ม parameter เพื่อรับชื่อ collection ที่ต้องการ
) -> Tuple[str, List[PointStruct]] | None:
    """
    Processes a single row from Google Sheet, performs embedding,
    and returns a tuple of (unique_id, list of PointStructs) for Qdrant upsert.
    This version processes the entire row as a single document.
    """
    unique_id_raw = row_data.get(GOOGLE_SHEET_UNIQUE_ID_COLUMN)
    last_updated_str = row_data.get(GOOGLE_SHEET_LAST_UPDATED_COLUMN)

    if not unique_id_raw:
        logging.warning(f"[{sheet_name}] Row {row_index+1} missing '{GOOGLE_SHEET_UNIQUE_ID_COLUMN}'. Skipping.")
        return None

    unique_id = str(unique_id_raw).strip()
    if not unique_id:
        logging.warning(f"[{sheet_name}] Row {row_index+1} has empty unique ID after stripping. Skipping.")
        return None

    # Parse datetime
    last_updated_sheet = None
    if last_updated_str:
        try:
            last_updated_sheet = datetime.datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
        except ValueError:
            try:
                last_updated_sheet = datetime.datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                logging.warning(f"[{sheet_name}] Invalid timestamp '{last_updated_str}' for row ID '{unique_id}'.")
    
    # Check if update needed
    try:
        existing_points, _ = await asyncio.to_thread(
            lambda: qdrant_client.scroll(
                collection_name=qdrant_target_collection, # ใช้ชื่อ collection ที่ส่งเข้ามา
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="google_sheet_row_id", 
                            match=MatchValue(value=unique_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        )
        if existing_points:
            last_updated_qdrant_str = existing_points[0].payload.get("last_updated_from_sheet")
            if last_updated_qdrant_str:
                try:
                    last_updated_qdrant = datetime.datetime.fromisoformat(last_updated_qdrant_str.replace("Z", "+00:00"))
                    if last_updated_sheet and last_updated_sheet <= last_updated_qdrant:
                        logging.info(f"[{sheet_name}] Row ID '{unique_id}' is up-to-date. Skipping.")
                        return None
                except ValueError:
                    logging.warning(f"[{sheet_name}] Invalid Qdrant timestamp for '{unique_id}'. Proceeding with update.")
    except UnexpectedResponse as e: # เปลี่ยน CollectionNotExist เป็น UnexpectedResponse
        if "Not found: Collection" in str(e): # ตรวจสอบข้อความ error
            logging.info(f"[{sheet_name}] Collection '{qdrant_target_collection}' does not exist yet. All rows will be treated as new.")
        else:
            logging.warning(f"[{sheet_name}] Unexpected Qdrant API response when checking for '{unique_id}': {e}. Proceeding with update.")
    except Exception as e:
        logging.warning(f"[{sheet_name}] Error checking Qdrant for '{unique_id}': {e}. Proceeding with update.")

    # Extract relevant fields
    account_id = row_data.get("ผังบัญชี")
    account_name = str(row_data.get("ชื่อผังบัญชี", "")).strip()
    description = str(row_data.get("คำอธิบายผังบัญชี", "")).strip()
    account_type = str(row_data.get("ประเภทผังบัญชี", "")).strip()

    content_text = (
        f"Source File: {sheet_name}\n"
        f"ผังบัญชี: {account_id}\n"
        f"ชื่อผังบัญชี: {account_name}\n"
        f"ประเภทผังบัญชี: {account_type}\n"
        f"คำอธิบายผังบัญชี: {description}"
    )

    if not content_text.strip():
        logging.warning(f"[{sheet_name}] No content for embedding in row ID '{unique_id}'. Skipping.")
        return None

    # Generate embedding for the entire content_text
    embedding_results = await async_generate_embedding([content_text]) 
    
    if not embedding_results or embedding_results[0] is None:
        logging.warning(f"[{sheet_name}] Embedding failed for row ID '{unique_id}'. Skipping.")
        return None
    
    embedding = embedding_results[0] 

    points_to_upsert: List[PointStruct] = []
    
    payload = {
        "google_sheet_row_id": unique_id,
        "last_updated_from_sheet": last_updated_str,
        "source_file": sheet_name,
        "page_number": None, 
        "content": content_text, 
        "metadata_flat": {
            "source_file": sheet_name,
            "ผังบัญชี": account_id,
            "ชื่อผังบัญชี": account_name,
            "คำอธิบายผังบัญชี": description,
            "ประเภทผังบัญชี": account_type,
        },
        "ผังบัญชี": account_id, 
        "ชื่อผังบัญชี": account_name,
        "คำอธิบายผังบัญชี": description,
        "ประเภทผังบัญชี": account_type,
        "batch": None, 
        "chunk_index": 0, 
    }

    points_to_upsert.append(
        PointStruct(
            id=str(uuid.uuid4()), 
            vector=embedding,
            payload=payload
        )
    )

    if points_to_upsert:
        logging.info(f"[{sheet_name}] Row ID '{unique_id}' prepared 1 point for upsert (full row).")
        return unique_id, points_to_upsert
    else:
        logging.info(f"[{sheet_name}] No points prepared for row ID '{unique_id}'.")
        return None


async def sync_google_sheet_to_qdrant(
    concurrency: int = 2,
    batch_size: int = 10
):
    """
    Main function to sync data from Google Sheet to Qdrant.
    It fetches data, checks for changes, processes new/modified rows,
    and updates Qdrant accordingly in batches.
    Each Google Sheet worksheet will be synchronized to its own Qdrant collection.
    """
    if not gc or not GOOGLE_SHEET_UNIQUE_ID_COLUMN or not GOOGLE_SHEET_LAST_UPDATED_COLUMN:
        logging.error("Google Sheet configuration is incomplete. Skipping sync.")
        return {"status": "failed", "detail": "Google Sheet configuration incomplete."}

    if not GOOGLE_SHEET_WORKSHEET_NAMES:
        logging.warning("No Google Sheet worksheet names configured. Nothing to synchronize.")
        return {"status": "success", "detail": "No Google Sheet worksheet names configured."}

    total_overall_chunks_upserted = 0
    total_overall_rows_processed = 0
    total_overall_base_ids_deleted = 0
    
    overall_sync_results = {}

    for sheet_name in GOOGLE_SHEET_WORKSHEET_NAMES:
        current_qdrant_collection = f"{QDRANT_COLLECTION_PREFIX}{sheet_name.lower().replace(' ', '_').replace('-', '_')}" 
        logging.info(f"--- Starting synchronization for worksheet: '{sheet_name}' to Qdrant collection: '{current_qdrant_collection}' ---")

        # สร้าง Collection หากยังไม่มี
        try:
            await asyncio.to_thread(
                lambda: qdrant_client.get_collection(collection_name=current_qdrant_collection)
            )
            logging.info(f"Qdrant collection '{current_qdrant_collection}' already exists.")
        except grpc.RpcError as e: # !!! เปลี่ยนตรงนี้ให้จับ grpc.RpcError ก่อน !!!
            if e.code() == grpc.StatusCode.NOT_FOUND:
                logging.info(f"Qdrant collection '{current_qdrant_collection}' not found (gRPC error). Creating it...")
                try:
                    await asyncio.to_thread(
                        lambda: qdrant_client.create_collection(
                            collection_name=current_qdrant_collection,
                            vectors_config=VectorParams(size=1024, distance=Distance.COSINE) # ขนาด vector ควรตรงกับ model embedding ที่ใช้
                        )
                    )
                    logging.info(f"Qdrant collection '{current_qdrant_collection}' created successfully.")
                except Exception as create_e:
                    logging.error(f"Error creating Qdrant collection '{current_qdrant_collection}': {create_e}", exc_info=True)
                    overall_sync_results[sheet_name] = {"status": "failed", "detail": f"Failed to create collection: {create_e}"}
                    continue # ข้ามชีทนี้ไป
            else: # ถ้าเป็น gRPC error อื่นๆ ที่ไม่ใช่ NOT_FOUND
                logging.error(f"gRPC error when checking collection '{current_qdrant_collection}': {e}", exc_info=True)
                overall_sync_results[sheet_name] = {"status": "failed", "detail": f"gRPC error checking collection: {e}"}
                continue
        except UnexpectedResponse as e: # ย้าย UnexpectedResponse มาเป็นลำดับที่สอง
            if "Not found: Collection" in str(e): # ตรวจสอบข้อความ error
                logging.info(f"Qdrant collection '{current_qdrant_collection}' not found (UnexpectedResponse). Creating it...")
                try:
                    await asyncio.to_thread(
                        lambda: qdrant_client.create_collection(
                            collection_name=current_qdrant_collection,
                            vectors_config=VectorParams(size=1024, distance=Distance.COSINE) # ขนาด vector ควรตรงกับ model embedding ที่ใช้
                        )
                    )
                    logging.info(f"Qdrant collection '{current_qdrant_collection}' created successfully.")
                except Exception as create_e:
                    logging.error(f"Error creating Qdrant collection '{current_qdrant_collection}': {create_e}", exc_info=True)
                    overall_sync_results[sheet_name] = {"status": "failed", "detail": f"Failed to create collection: {create_e}"}
                    continue # ข้ามชีทนี้ไป
            else: # ถ้าเป็น UnexpectedResponse แต่อ้างอิงถึง error อื่น
                logging.error(f"Unexpected Qdrant API response when checking collection '{current_qdrant_collection}': {e}", exc_info=True)
                overall_sync_results[sheet_name] = {"status": "failed", "detail": f"Qdrant API error: {e}"}
                continue
        except Exception as e: # Catch all other exceptions during get_collection
            logging.error(f"General error checking Qdrant collection '{current_qdrant_collection}': {e}", exc_info=True)
            overall_sync_results[sheet_name] = {"status": "failed", "detail": f"General error checking collection: {e}"}
            continue

        # ... (ส่วนที่เหลือของโค้ด)

        sheet_data = await get_google_sheet_data(sheet_name)
        if not sheet_data:
            logging.info(f"No data found in worksheet '{sheet_name}'. Skipping sync for this sheet.")
            overall_sync_results[sheet_name] = {"status": "success", "detail": "No data found."}
            continue

        existing_qdrant_base_ids = set()
        try:
            # ใช้ collection_name ที่ถูกต้องในการ scroll
            scroll_result, _ = await asyncio.to_thread(
                lambda: qdrant_client.scroll(
                    collection_name=current_qdrant_collection,
                    limit=10000, 
                    with_payload=True,
                    with_vectors=False,
                )
            )
            for point in scroll_result:
                if point.payload and "google_sheet_row_id" in point.payload:
                    existing_qdrant_base_ids.add(point.payload["google_sheet_row_id"])

            logging.info(f"Found {len(existing_qdrant_base_ids)} unique base IDs in Qdrant collection '{current_qdrant_collection}'.")
        except UnexpectedResponse as e: # เปลี่ยน CollectionNotExist เป็น UnexpectedResponse
            if "Not found: Collection" in str(e): # ตรวจสอบข้อความ error
                logging.info(f"Collection '{current_qdrant_collection}' not found during scroll. No existing IDs to check.")
                existing_qdrant_base_ids = set() # ตั้งค่าให้เป็น set ว่างเปล่า
            else:
                logging.error(f"Unexpected Qdrant API response when fetching existing IDs for '{current_qdrant_collection}': {e}. Cannot reliably detect deletions for this sheet.", exc_info=True)
                existing_qdrant_base_ids = set() # ตั้งค่าให้เป็น set ว่างเปล่าเพื่อป้องกันการทำงานผิดพลาด
        except Exception as e:
            logging.error(f"Error fetching existing Qdrant base IDs for '{current_qdrant_collection}': {e}. Cannot reliably detect deletions for this sheet.", exc_info=True)


        sheet_unique_ids = set()
        processing_tasks = []
        semaphore = asyncio.Semaphore(concurrency) # Concurrency for sheet row processing

        for i, row in enumerate(sheet_data):
            unique_id = row.get(GOOGLE_SHEET_UNIQUE_ID_COLUMN)
            if unique_id:
                sheet_unique_ids.add(str(unique_id).strip()) 
                task = asyncio.create_task(
                    process_google_sheet_row_for_embedding(
                        row_data=row,
                        row_index=i,
                        sheet_name=row.get("__worksheet__", ""),
                        qdrant_target_collection=current_qdrant_collection # ส่งชื่อ collection ไปยังฟังก์ชันประมวลผลแถว
                    )
                )
                processing_tasks.append(task)
            else:
                logging.warning(f"[{sheet_name}] Row {i+1} skipped due to missing '{GOOGLE_SHEET_UNIQUE_ID_COLUMN}'.")

        # รอให้ทุก task ของการประมวลผลแถวเสร็จสิ้น
        processed_results = await asyncio.gather(*processing_tasks)
        
        # กรองเฉพาะผลลัพธ์ที่ไม่ใช่ None และเป็น (unique_id, List[PointStruct])
        all_points_to_upsert: List[PointStruct] = []
        sheet_chunks_upserted = 0
        sheet_rows_processed = 0

        for res in processed_results:
            if res is not None:
                unique_id, points = res
                all_points_to_upsert.extend(points)
                sheet_chunks_upserted += len(points) 
                sheet_rows_processed += 1

        # แบ่ง all_points_to_upsert เป็น batch และ upsert
        if all_points_to_upsert:
            logging.info(f"Prepared {len(all_points_to_upsert)} total points for upserting in batches to '{current_qdrant_collection}'.")
            for i in range(0, len(all_points_to_upsert), batch_size):
                batch = all_points_to_upsert[i : i + batch_size]
                await save_points_to_qdrant(batch, collection_name=current_qdrant_collection)
                logging.info(f"Upserted batch {int(i/batch_size) + 1}/{(len(all_points_to_upsert) + batch_size - 1) // batch_size} ({len(batch)} points) to '{current_qdrant_collection}'.")
        else:
            logging.info(f"No new or updated points to upsert for collection '{current_qdrant_collection}'.")

        # ลบจุดที่ไม่มีใน Google Sheet แล้วออกจาก Qdrant collection ปัจจุบัน
        ids_to_delete_from_qdrant = existing_qdrant_base_ids - sheet_unique_ids
        sheet_base_ids_deleted = 0
        if ids_to_delete_from_qdrant:
            logging.info(f"Detected {len(ids_to_delete_from_qdrant)} rows deleted from Google Sheet '{sheet_name}'. Deleting corresponding points from Qdrant collection '{current_qdrant_collection}'.")
            
            for base_id_to_delete in ids_to_delete_from_qdrant:
                try:
                    await asyncio.to_thread(
                        lambda: qdrant_client.delete_points(
                            collection_name=current_qdrant_collection,
                            points_selector=FilterSelector(
                                filter=Filter(
                                    must=[
                                        FieldCondition(
                                            key="google_sheet_row_id", 
                                            match=MatchValue(value=base_id_to_delete)
                                        )
                                    ]
                                )
                            )
                        )
                    )
                    logging.info(f"Successfully deleted all points for base ID '{base_id_to_delete}' from '{current_qdrant_collection}'.")
                    sheet_base_ids_deleted += 1
                except Exception as e:
                    logging.error(f"Error deleting points for base ID '{base_id_to_delete}' from '{current_qdrant_collection}': {e}", exc_info=True)
        else:
            logging.info(f"No rows detected as deleted from Google Sheet '{sheet_name}'.")

        total_overall_chunks_upserted += sheet_chunks_upserted
        total_overall_rows_processed += sheet_rows_processed
        total_overall_base_ids_deleted += sheet_base_ids_deleted

        overall_sync_results[sheet_name] = {
            "status": "success",
            "total_rows_in_sheet": len(sheet_data),
            "total_rows_processed_or_updated": sheet_rows_processed,
            "total_qdrant_chunks_upserted": sheet_chunks_upserted,
            "total_qdrant_base_ids_deleted": sheet_base_ids_deleted,
            "qdrant_collection_name": current_qdrant_collection,
        }
        logging.info(f"--- Synchronization for worksheet '{sheet_name}' complete. ---")

    logging.info("Google Sheet to Qdrant synchronization complete for all configured worksheets.")

    return {
        "status": "success",
        "detail": "Google Sheet synchronization completed for all configured worksheets.",
        "summary_per_sheet": overall_sync_results,
        "overall_summary": {
            "total_worksheets_synced": len(GOOGLE_SHEET_WORKSHEET_NAMES),
            "total_overall_rows_processed": total_overall_rows_processed,
            "total_overall_qdrant_chunks_upserted": total_overall_chunks_upserted,
            "total_overall_qdrant_base_ids_deleted": total_overall_base_ids_deleted,
            "embedding_model_used": "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest", 
        }
    }

# --- FastAPI Application ---
app = FastAPI(
    title="Google Sheet RAG Bot Sync Service",
    description="Synchronizes text data from Google Sheet to Qdrant with Embedding via local Ollama/Qwen. Supports per-sheet collection.",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Google Sheet RAG Bot Sync Service is running! Visit /docs for API documentation."}

@app.post("/sync/google_sheet_to_qdrant")
async def trigger_google_sheet_sync(
    concurrency_sheet_rows: int = 2,
    qdrant_batch_size: int = 10
) -> JSONResponse:
    """
    Triggers the synchronization process from Google Sheet to Qdrant.
    It fetches data, checks for changes, processes new/modified rows,
    and updates Qdrant accordingly, with each worksheet syncing to its own Qdrant collection.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured. Please check server settings."
        )
    
    # ควรมีเช็คว่า Ollama/Qwen endpoint ใช้งานได้จริงหรือไม่ แทน JINA_API_KEY
    # ตรวจสอบว่า Qdrant client พร้อมใช้งาน
    if qdrant_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Qdrant client is not initialized. Please check QDRANT_URL and QDRANT_API_KEY."
        )

    # ทดสอบการเชื่อมต่อกับ Ollama/Qwen Embedding Service
    embedding_service_url = "http://192.168.1.10:11434/api/embeddings" # ควรเปลี่ยนเป็น Env Var
    try:
        response = await asyncio.to_thread(requests.get, embedding_service_url.replace("/api/embeddings", "/")) # ลองเรียก endpoint พื้นฐานของ Ollama
        response.raise_for_status()
        if "Ollama" not in response.text: # ตรวจสอบว่าเป็น Ollama หรือไม่ (อาจจะปรับตาม response จริง)
             logging.warning(f"Embedding service at {embedding_service_url} might not be Ollama or not configured correctly.")
        logging.info("Successfully connected to local embedding service (Ollama/Qwen).")
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Local embedding service (Ollama/Qwen) at {embedding_service_url} is not reachable. Please ensure it is running."
        )
    except Exception as e:
        logging.error(f"Error checking local embedding service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking local embedding service: {e}"
        )


    if not gc or not GOOGLE_SHEET_ID or not GOOGLE_SHEET_WORKSHEET_NAMES or \
       not GOOGLE_SHEET_UNIQUE_ID_COLUMN or not GOOGLE_SHEET_LAST_UPDATED_COLUMN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google Sheet configuration is incomplete. Please check GOOGLE_SHEET_KEY_PATH, GOOGLE_SHEET_ID, GOOGLE_SHEET_WORKSHEET_NAME, GOOGLE_SHEET_UNIQUE_ID_COLUMN, and GOOGLE_SHEET_LAST_UPDATED_COLUMN in your environment variables."
        )

    try:
        sync_summary = await sync_google_sheet_to_qdrant(
            concurrency=concurrency_sheet_rows,
            batch_size=qdrant_batch_size
        )
        return JSONResponse(
            content=sync_summary,
            status_code=status.HTTP_200_OK
        )
    except HTTPException:
        raise # Re-raise HTTPExceptions that were already caught and handled
    except Exception as e:
        logging.error(f"Error during Google Sheet sync operation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred during Google Sheet synchronization: {e}"
        )