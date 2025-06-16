import os
import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse 
import uuid

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


# --- Jina AI Configuration ---
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-clip-v2")

if not JINA_API_KEY:
    logging.error("JINA_API_KEY not found. Jina AI embedding generation will not work.")
else:
    logging.info("Jina AI API configured successfully.")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

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

        # สร้าง collection หากยังไม่มี (บล็อกนี้คือส่วนที่แก้ไข)
        try:
            # ลองตรวจสอบว่า collection มีอยู่แล้วหรือไม่
            collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
            logging.info(f"Qdrant collection '{QDRANT_COLLECTION}' already exists. Using existing collection.")

        except Exception as e: # <<< เปลี่ยนมาจับ Exception ตัวนี้ และย้าย logic ทั้งหมดมาไว้ที่นี่
            # ตรวจสอบข้อความ "Not found" ใน error message จาก _InactiveRpcError หรือ UnexpectedResponse
            if "Not found: Collection" in str(e) or "Not found" in str(e):
                logging.info(f"Qdrant collection '{QDRANT_COLLECTION}' not found. Creating it...")
                try:
                    qdrant_client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                    )
                    logging.info(f"Qdrant collection '{QDRANT_COLLECTION}' created successfully.")
                except Exception as create_e:
                    logging.error(f"Error creating Qdrant collection '{QDRANT_COLLECTION}': {create_e}", exc_info=True)
                    exit(1)
            else:
                # ถ้าไม่ใช่ error "Not found" ให้แจ้งว่าเป็น error อื่น
                logging.error(f"Unexpected error checking Qdrant collection '{QDRANT_COLLECTION}': {e}", exc_info=True)
                exit(1)
    except Exception as e:
        logging.error(f"Error initializing Qdrant client with cloud URL: {e}", exc_info=True) # เพิ่ม exc_info=True
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
worksheets = []

if GOOGLE_SHEET_KEY_PATH and GOOGLE_SHEET_ID and GOOGLE_SHEET_WORKSHEET_NAMES:
    try:
        creds = Credentials.from_service_account_file(
            GOOGLE_SHEET_KEY_PATH,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GOOGLE_SHEET_ID)
        logging.info("Google Sheet client authorized successfully.")

        for sheet_name in GOOGLE_SHEET_WORKSHEET_NAMES:
            try:
                worksheet = sh.worksheet(sheet_name.strip())  # ใส่ strip() เผื่อลบช่องว่างรอบๆ
                worksheets.append((sheet_name.strip(), worksheet))  # append เข้า list ได้ถูกต้อง
                logging.info(f"Loaded worksheet '{sheet_name.strip()}' successfully.")
            except Exception as ws_error:
                logging.error(f"Failed to load worksheet '{sheet_name.strip()}': {ws_error}")
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

    url = "http://192.168.1.10:11434/api/embeddings"
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
    collection_name: str = QDRANT_COLLECTION
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
        logging.info(f"✅ Qdrant upserted {len(points)} points successfully.")
    except Exception as e:
        logging.error(
            f"❌ Error saving batch to Qdrant: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during Qdrant batch save: {e}"
        )

# --- Google Sheet Processing Functions ---
async def get_google_sheet_data() -> List[Dict[str, Any]]:
    """
    Connects to the Google Sheet and retrieves all data from configured worksheets
    as a list of dictionaries. Each dictionary represents a row with column headers as keys.
    """
    if gc is None:
        logging.error("Google Sheet client is not initialized. Cannot retrieve data.")
        return []

    if not GOOGLE_SHEET_ID or not GOOGLE_SHEET_WORKSHEET_NAMES:
        logging.error("Google Sheet ID or Worksheet Names are not configured. Cannot retrieve data.")
        return []

    try:
        spreadsheet = await asyncio.to_thread(gc.open_by_key, GOOGLE_SHEET_ID)
        all_data: List[Dict[str, Any]] = []

        for sheet_name in GOOGLE_SHEET_WORKSHEET_NAMES:
            try:
                worksheet = await asyncio.to_thread(spreadsheet.worksheet, sheet_name)
                data = await asyncio.to_thread(worksheet.get_all_records, head=2)
                logging.info(f"Retrieved {len(data)} rows from worksheet '{sheet_name}'.")
                
                # Optional: Add sheet name to each row for context
                for row in data:
                    row["__worksheet__"] = sheet_name

                all_data.extend(data)
            except gspread.exceptions.WorksheetNotFound:
                logging.warning(f"Worksheet '{sheet_name}' not found in spreadsheet '{GOOGLE_SHEET_ID}'. Skipping.")
            except Exception as sheet_error:
                logging.error(f"Error reading worksheet '{sheet_name}': {sheet_error}", exc_info=True)

        logging.info(f"Total combined rows from all worksheets: {len(all_data)}")
        return all_data

    except gspread.exceptions.SpreadsheetNotFound:
        logging.error(f"Google Sheet with ID '{GOOGLE_SHEET_ID}' not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Google Sheet with ID '{GOOGLE_SHEET_ID}' not found.")
    except Exception as e:
        logging.error(f"Error fetching data from Google Sheet: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching data from Google Sheet: {e}")

async def process_google_sheet_row_for_embedding(
    row_data: Dict[str, Any],
    row_index: int,
    sheet_name: str, # sheet_name ถูกส่งเข้ามาแล้ว
) -> Tuple[str, List[PointStruct]] | None:
    """
    Processes a single row from Google Sheet, performs embedding,
    and returns a tuple of (unique_id, list of PointStructs) for Qdrant upsert.
    This version processes the entire row as a single document.
    """
    unique_id_raw = row_data.get(GOOGLE_SHEET_UNIQUE_ID_COLUMN)
    last_updated_str = row_data.get(GOOGLE_SHEET_LAST_UPDATED_COLUMN)

    if not unique_id_raw:
        logging.warning(f"Row {row_index+1} missing '{GOOGLE_SHEET_UNIQUE_ID_COLUMN}'. Skipping.")
        return None

    unique_id = str(unique_id_raw).strip()
    if not unique_id:
        logging.warning(f"Row {row_index+1} has empty unique ID after stripping. Skipping.")
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
                logging.warning(f"Invalid timestamp '{last_updated_str}' for row ID '{unique_id}'.")
    
    # Check if update needed
    should_update = True
    try:
        existing_points, _ = await asyncio.to_thread(
            lambda: qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter={"must": [{"key": "google_sheet_row_id", "match": {"value": unique_id}}]},
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
                        logging.info(f"Row ID '{unique_id}' is up-to-date. Skipping.")
                        return None
                except ValueError:
                    logging.warning(f"Invalid Qdrant timestamp for '{unique_id}'. Proceeding with update.")
    except Exception as e:
        logging.warning(f"Error checking Qdrant for '{unique_id}': {e}. Proceeding with update.")

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
        logging.warning(f"No content for embedding in row ID '{unique_id}'. Skipping.")
        return None

    # Generate embedding for the entire content_text
    embedding_results = await async_generate_embedding([content_text]) 
    
    if not embedding_results or embedding_results[0] is None:
        logging.warning(f"Embedding failed for row ID '{unique_id}'. Skipping.")
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
        logging.info(f"Row ID '{unique_id}' prepared 1 point for upsert (full row).")
        return unique_id, points_to_upsert
    else:
        logging.info(f"No points prepared for row ID '{unique_id}'.")
        return None


async def sync_google_sheet_to_qdrant(
    # Removed chunk_size and overlap parameters as they are no longer used for row-based embedding
    concurrency: int = 2,
    batch_size: int = 10
):
    """
    Main function to sync data from Google Sheet to Qdrant.
    It fetches data, checks for changes, processes new/modified rows,
    and updates Qdrant accordingly in batches.
    """
    if not gc or not GOOGLE_SHEET_UNIQUE_ID_COLUMN or not GOOGLE_SHEET_LAST_UPDATED_COLUMN:
        logging.error("Google Sheet configuration is incomplete. Skipping sync.")
        return {"status": "failed", "detail": "Google Sheet configuration incomplete."}

    try:
        # Removed gemini_text_model initialization as it's not directly used for embedding in this flow
        pass
    except Exception as e:
        logging.error(f"Failed to load Gemini GenerativeModel for Google Sheet sync: {e}")
        # This error is now less critical as Gemini isn't directly used for embedding
        # You might choose to log and continue, or raise, depending on your application's needs.
        # For now, keeping the raise to indicate potential issues if Gemini was intended for other tasks.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load Gemini GenerativeModel (if needed for other tasks): {e}"
        )

    logging.info("Starting Google Sheet to Qdrant synchronization.")
    sheet_data = await get_google_sheet_data()
    if not sheet_data:
        return {"status": "success", "detail": "No data found in Google Sheet to synchronize."}

    existing_qdrant_base_ids = set()
    try:
        scroll_result, _ = await asyncio.to_thread(
            lambda: qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=10000, # Increased limit to fetch more existing IDs
                with_payload=True,
                with_vectors=False,
            )
        )
        for point in scroll_result:
            if point.payload and "google_sheet_row_id" in point.payload:
                existing_qdrant_base_ids.add(point.payload["google_sheet_row_id"])

        logging.info(f"Found {len(existing_qdrant_base_ids)} unique base IDs in Qdrant.")
    except Exception as e:
        logging.error(f"Error fetching existing Qdrant base IDs: {e}. Cannot reliably detect deletions.", exc_info=True)

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
                    # Removed chunk_size and overlap from call
                )
            )
            processing_tasks.append(task)
        else:
            logging.warning(f"Row {i+1} skipped due to missing '{GOOGLE_SHEET_UNIQUE_ID_COLUMN}'.")

    # รอให้ทุก task ของการประมวลผลแถวเสร็จสิ้น
    processed_results = await asyncio.gather(*processing_tasks)
    
    # กรองเฉพาะผลลัพธ์ที่ไม่ใช่ None และเป็น (unique_id, List[PointStruct])
    all_points_to_upsert: List[PointStruct] = []
    total_overall_chunks_upserted = 0
    for res in processed_results:
        if res is not None:
            unique_id, points = res
            all_points_to_upsert.extend(points)
            total_overall_chunks_upserted += len(points) # นับจำนวน chunks ที่จะ upsert (ตอนนี้คือ 1 ต่อแถว)

    # แบ่ง all_points_to_upsert เป็น batch และ upsert
    logging.info(f"Prepared {len(all_points_to_upsert)} total points for upserting in batches.")
    for i in range(0, len(all_points_to_upsert), batch_size):
        batch = all_points_to_upsert[i : i + batch_size]
        await save_points_to_qdrant(batch, collection_name=QDRANT_COLLECTION)
        logging.info(f"Upserted batch {int(i/batch_size) + 1}/{(len(all_points_to_upsert) + batch_size - 1) // batch_size} ({len(batch)} points).")

    # ลบจุดที่ไม่มีใน Google Sheet แล้วออกจาก Qdrant
    ids_to_delete_from_qdrant = existing_qdrant_base_ids - sheet_unique_ids
    if ids_to_delete_from_qdrant:
        logging.info(f"Detected {len(ids_to_delete_from_qdrant)} rows deleted from Google Sheet. Deleting corresponding points from Qdrant.")
        
        for base_id_to_delete in ids_to_delete_from_qdrant:
            try:
                await asyncio.to_thread(
                    lambda: qdrant_client.delete_points(
                        collection_name=QDRANT_COLLECTION,
                        points_selector={"filter": {"must": [{"key": "google_sheet_row_id", "match": {"value": base_id_to_delete}}]}}
                    )
                )
                logging.info(f"Successfully deleted all points for base ID '{base_id_to_delete}' from Qdrant.")
            except Exception as e:
                logging.error(f"Error deleting points for base ID '{base_id_to_delete}' from Qdrant: {e}", exc_info=True)
    else:
        logging.info("No rows detected as deleted from Google Sheet.")

    total_overall_prompt_tokens = 0 
    total_overall_candidates_tokens = 0 

    logging.info("Google Sheet to Qdrant synchronization complete.")

    return {
        "status": "success",
        "detail": "Google Sheet synchronization completed.",
        "summary": {
            "total_rows_in_sheet": len(sheet_data),
            "total_rows_processed_or_updated": len([res for res in processed_results if res is not None]),
            "total_qdrant_chunks_upserted": total_overall_chunks_upserted,
            "total_qdrant_base_ids_deleted": len(ids_to_delete_from_qdrant),
            "overall_gemini_prompt_tokens": total_overall_prompt_tokens,
            "overall_gemini_candidate_tokens": total_overall_candidates_tokens,
            "embedding_model_used": "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest", # Hardcoded based on async_generate_embedding
        }
    }


# --- FastAPI Application ---
app = FastAPI(
    title="Google Sheet RAG Bot Sync Service",
    description="Synchronizes text data from Google Sheet to Qdrant with Embedding via Jina AI.",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Google Sheet RAG Bot Sync Service is running! Visit /docs for API documentation."}

@app.post("/sync/google_sheet_to_qdrant")
async def trigger_google_sheet_sync(
    # Removed chunk_size and overlap parameters from API endpoint
    concurrency_sheet_rows: int = 2,
    qdrant_batch_size: int = 10
) -> JSONResponse:
    """
    Triggers the synchronization process from Google Sheet to Qdrant.
    It fetches data, checks for changes, processes new/modified rows,
    and updates Qdrant accordingly.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured. Please check server settings."
        )
    if not JINA_API_KEY: # This check might be misleading now if Jina is not used for embedding. Consider clarifying or removing if not relevant.
        logging.warning("JINA_API_KEY is present but Jina AI embedding is not directly used in the current setup. Embedding relies on http://192.168.1.10:11434/api/embeddings.")
        # Decide if you want to raise an HTTPException here, or just log a warning.
        # For now, keeping it as an HTTPException to align with the original intent of having an embedding service.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding service (local Ollama/Qwen) not reachable or misconfigured. Please check the embedding API URL."
        )


    if not gc or not GOOGLE_SHEET_ID or not GOOGLE_SHEET_WORKSHEET_NAMES or \
       not GOOGLE_SHEET_UNIQUE_ID_COLUMN or not GOOGLE_SHEET_LAST_UPDATED_COLUMN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google Sheet configuration is incomplete. Please check GOOGLE_SHEET_KEY_PATH, GOOGLE_SHEET_ID, GOOGLE_SHEET_WORKSHEET_NAME, GOOGLE_SHEET_UNIQUE_ID_COLUMN, and GOOGLE_SHEET_LAST_UPDATED_COLUMN in your environment variables."
        )

    try:
        sync_summary = await sync_google_sheet_to_qdrant(
            # Removed chunk_size and overlap from call
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