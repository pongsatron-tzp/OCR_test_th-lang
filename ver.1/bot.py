import os
import logging
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

import json

logging.basicConfig(level=logging.INFO)
load_dotenv("key.env")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logging.error("❌ Missing GEMINI_API_KEY in environment variables")
    raise SystemExit("Missing GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
app = FastAPI()

def get_embedding(text):
    try:
        url = "http://192.168.1.10:11434/api/embeddings"
        payload = {
            "model": "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:latest",
            "prompt": text
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if 'embedding' in data and isinstance(data['embedding'], list):
            embedding = data['embedding']
            logging.info(f"✅ API OK: embedding length {len(embedding)}")
            return embedding
        else:
            logging.error(f"❌ API Response ไม่ถูกต้อง: {data}")
            return None
    except Exception as e:
        logging.error(f"❌ Embedding error: {e}")
        return None

def extract_metadata_filter_with_llm(query: str):
    prompt = f"""
คุณคือผู้ช่วยวิเคราะห์คำถามภาษาไทย
ให้ดึงข้อมูลประเภท metadata filter ที่เหมาะสมในรูปแบบ JSON จากคำถามนี้

คำถาม: {query}

ขอให้ตอบกลับเป็น JSON เท่านั้น โดย key เป็นชื่อฟิลด์ metadata และ value เป็นค่าที่ควรใช้กรอง

ถ้าไม่พบ metadata ให้ตอบเป็น JSON ว่าง: {{}}
"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 128,
                "temperature": 0,
            }
        )
        data = json.loads(response.text.strip())
        if isinstance(data, dict) and data:
            return data
    except Exception as e:
        logging.error(f"❌ LLM extract metadata filter error: {e}")
    return None

def build_metadata_filter(metadata_dict):
    if not metadata_dict:
        return None

    must_conditions = []
    for key, value in metadata_dict.items():
        if value:
            # ในที่นี้สมมติว่า value เป็น string หรือ number ตรงๆ
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchAny(any=[str(value)])
                )
            )
    if not must_conditions:
        return None

    return Filter(must=must_conditions)

def search_documents(query_vector, collection, limit=40, exclude_ids=None, metadata_filter=None):
    try:
        filter_condition = metadata_filter

        if exclude_ids:
            if filter_condition is None:
                filter_condition = Filter(
                    must_not=[
                        FieldCondition(
                            key="ผังบัญชี",
                            match=MatchAny(any=[str(eid) for eid in exclude_ids])
                        )
                    ]
                )
            else:
                # ถ้ามี filter อยู่แล้ว ให้ต่อ must_not เข้าไปด้วย
                filter_condition.must_not = filter_condition.must_not or []
                filter_condition.must_not.append(
                    FieldCondition(
                        key="ผังบัญชี",
                        match=MatchAny(any=[str(eid) for eid in exclude_ids])
                    )
                )

        results = qdrant_client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=filter_condition
        )
        return results
    except Exception as e:
        logging.error(f"❌ Qdrant search error: {e}")
        return []

def format_context(hit):
    account_id = hit.payload.get('ผังบัญชี', '')
    account_name = hit.payload.get('ชื่อผังบัญชี', '')
    account_type = hit.payload.get('ประเภทผังบัญชี', '')
    description = hit.payload.get('คำอธิบายผังบัญชี', '')
    return (
        f"ผังบัญชี: {account_id}, "
        f"ชื่อผังบัญชี: {account_name}, "
        f"ประเภทผังบัญชี: {account_type}, "
        f"คำอธิบายผังบัญชี: {description}"
    )

def llm_fn(query, contexts):
    prompt = f"""
คุณคือผู้ช่วยผู้เชี่ยวชาญด้านผังบัญชี
คำถาม: {query}

ข้อมูลที่เกี่ยวข้อง (ใช้ข้อมูลนี้ในการวิเคราะห์และตอบคำถามอย่างละเอียด):
{chr(10).join(contexts)}

กรุณาวิเคราะห์ข้อมูลอย่างรอบคอบและตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้เท่านั้น
ถ้าไม่พบข้อมูลหรือไม่แน่ใจ ให้ตอบว่า 'ไม่พบข้อมูลที่ชัดเจน' หรือ 'ไม่สามารถตอบคำถามนี้ได้จากข้อมูลที่มี'

ตอบเป็นภาษาไทย และให้คำตอบที่กระชับ ชัดเจน และตรงประเด็น:
"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 512,
                "temperature": 0.5,
            }
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลคำถามจาก Gemini API (SDK): {e}")
        return "เกิดข้อผิดพลาดในการประมวลผลคำถาม"

def rerank_by_relevance(query, contexts):
    prompt = f"""
คุณคือผู้ช่วยจัดอันดับข้อมูลที่มีความเข้าใจลึกซึ้งในเรื่องผังบัญชีและสามารถวิเคราะห์ความเกี่ยวข้องของข้อมูลกับคำถามได้อย่างละเอียด

คำถาม: {query}

โปรดพิจารณาข้อมูลต่อไปนี้แต่ละรายการ ว่ามี "ความเกี่ยวข้องโดยตรง" หรือ "ความเกี่ยวข้องสูง" กับคำถามหรือไม่:
{chr(10).join(f"{i+1}. {ctx}" for i, ctx in enumerate(contexts))}

วิธีการ:
- ให้วิเคราะห์เนื้อหาของแต่ละข้อมูลอย่างละเอียด
- ให้พิจารณาความหมาย, คำสำคัญ, หมวดหมู่ หรือรายละเอียดที่ตรงกับคำถาม
- ให้จัดอันดับข้อมูลจากความเกี่ยวข้องมากที่สุดไปน้อยที่สุด

รูปแบบคำตอบ:
กรุณาตอบกลับโดยระบุลำดับหมายเลขที่จัดอันดับไว้ โดยใช้รูปแบบนี้เท่านั้น: 2,1,4,3,5

ห้ามให้คำอธิบายเพิ่มเติม

ถ้าพร้อมแล้ว กรุณาเรียงลำดับข้อมูลให้เลย:
"""
    model = genai.GenerativeModel("gemini-1.5-pro")
    try:
        response = model.generate_content(prompt)
        order = [int(i.strip()) - 1 for i in response.text.strip().split(",") if i.strip().isdigit()]
        return [contexts[i] for i in order if i < len(contexts)]
    except Exception as e:
        logging.error(f"❌ Rerank error: {e}")
        return contexts

def process_search_and_answer(query, collection, top_k=40, exclude_ids=None):
    query_vector = get_embedding(query)
    if not query_vector:
        return None, []

    metadata_dict = extract_metadata_filter_with_llm(query)
    metadata_filter = build_metadata_filter(metadata_dict)

    results = search_documents(query_vector, collection, limit=top_k, exclude_ids=exclude_ids, metadata_filter=metadata_filter)

    if not results:
        return None, []

    context_with_scores = [
        (format_context(hit), hit.score if hasattr(hit, "score") else 0.0)
        for hit in results
    ]

    context_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_contexts_for_rerank = [ctx for ctx, _ in context_with_scores[:15]]

    reranked_contexts = rerank_by_relevance(query, top_contexts_for_rerank)
    top_contexts = reranked_contexts[:8]

    answer = llm_fn(query, top_contexts)

    hit_ids = [hit.payload.get('ผังบัญชี', None) for hit in results]

    return answer, hit_ids

def agent(query, collection, top_k=40):
    answer, first_result_ids = process_search_and_answer(query, collection, top_k)

    if answer and "ไม่พบ" not in answer.lower() and len(answer) >= 20:
        return answer

    logging.info("🔁 ไม่พบข้อมูลชัดเจนในรอบแรก กำลังค้นหาข้อมูลใหม่ในรอบที่ 2 ...")

    answer, _ = process_search_and_answer(query, collection, top_k, exclude_ids=first_result_ids)

    if not answer or "ไม่พบ" in answer.lower() or len(answer) < 20:
        return "ไม่พบข้อมูลที่ชัดเจนในข้อมูลที่มี หรือกรุณาระบุคำถามให้ชัดเจนขึ้น (หลังค้นหา 2 รอบ)"

    return answer

@app.post("/webhook")
async def n8n_webhook(request: Request):
    data = await request.json()
    user_message = data.get("text", "").strip()

    logging.info(f"📩 Message from n8n webhook: {user_message}")

    if user_message.lower() == "/start":
        welcome_message = (
            "สวัสดีครับ! ผมคือผู้ช่วยด้านผังบัญชี\n"
            "ส่งคำถามของคุณมาได้เลยครับ เช่น พิมพ์คำถามหรือรายละเอียดที่ต้องการทราบ"
        )
        return JSONResponse({"answer": welcome_message})

    answer = agent(user_message, QDRANT_COLLECTION)
    return JSONResponse({"answer": answer})
