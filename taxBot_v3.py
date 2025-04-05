import gradio as gr
import pandas as pd
import json
import os
import fitz  # PyMuPDF 讀取 PDF
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from mistralai import Mistral
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # ✅ 加入進度條
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ 加入嵌入轉換器


# ✅ 確保安裝 openpyxl 以支援 Excel 讀取
try:
    import openpyxl
except ImportError:
    print("⚠️ 缺少 openpyxl，正在安裝...")
    os.system("pip install openpyxl")

# ✅ 設定檔案路徑
EXCEL_FILE = "/Users/yun/Downloads/tariff_data_2025/trade_tariff_database_202500.xlsx"
PDF_FILES = ["/Users/yun/Downloads/td-codes.pdf", "/Users/yun/Downloads/finalCopy.pdf", "/Users/yun/Downloads/td-fields.pdf"]
JSON_PATH = "trade_tariff_database.json"
CONFIG_PATH = "config.json"

# ✅ 讀取 API Key 來自 config.json
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
        MISTRAL_API_KEY = config.get("MISTRAL_API_KEY", "")
except Exception as e:
    MISTRAL_API_KEY = ""
    print(f"⚠️ 無法讀取 API Key: {str(e)}")

# ✅ 讀取 Excel 並轉換為 JSON
def load_excel_to_json(file_path):
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        if df.empty:
            print("⚠️ Excel 文件為空，請檢查數據來源。")
            return []
        records = df.to_dict(orient='records')
        return records
    except Exception as e:
        print(f"⚠️ 無法讀取 Excel 文件: {str(e)}")
        return []

# ✅ 讀取多個 PDF 檔案
def load_pdfs(pdf_files):
    pdf_texts = []
    for pdf_file in pdf_files:
        try:
            with fitz.open(pdf_file) as doc:
                text = "\n".join([page.get_text("text") for page in doc])
                pdf_texts.append(text)
        except Exception as e:
            print(f"⚠️ 無法讀取 PDF {pdf_file}: {str(e)}")
    return pdf_texts

# ✅ 加載並處理數據
database = load_excel_to_json(EXCEL_FILE)
pdf_data = load_pdfs(PDF_FILES)
combined_data = database + [{"content": text} for text in pdf_data]
if not combined_data:
    raise ValueError("⚠️ 無有效數據，請檢查 Excel 和 PDF 檔案")

# ✅ 嵌入模型 + FAISS 建立向量資料庫
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_data = [record.get("content", "") for record in combined_data]
text_splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text("\n".join(text_data))

embedded_vectors = []
for chunk in tqdm(text_splits, desc="🔍 正在生成嵌入向量"):
    embedded_vectors.append(embedding_model.embed_query(chunk))

text_embeddings = list(zip(text_splits, embedded_vectors))
vector_store = FAISS.from_embeddings(text_embeddings, embedding=embedding_model)

# ✅ 構建 BM25 關鍵字檢索
bm25_corpus = [record.get("content", "") for record in combined_data if "content" in record]
bm25 = BM25Okapi([desc.split() for desc in bm25_corpus if desc.strip()]) if bm25_corpus else None

def retrieve_relevant_records(query):
    keyword_matches = []
    if bm25:
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indexes = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
        keyword_matches = [combined_data[i] for i in top_indexes]

    vector_matches = []
    if vector_store:
        query_vector = embedding_model.embed_query(query)
        faiss_results = vector_store.similarity_search_by_vector(query_vector, k=5)
        for doc in faiss_results:
            vector_matches.append({"content": doc.page_content})

    def json_serializable(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return str(obj)

    results = list({json.dumps(r, ensure_ascii=False, default=json_serializable) for r in (keyword_matches + vector_matches)})
    return [json.loads(r) for r in results] if results else [{"message": "未找到相關資訊"}]

def chatbot_response(message, history):
    retrieved_data = retrieve_relevant_records(message)
    if not MISTRAL_API_KEY:
        return history.append({"role": "assistant", "content": "⚠️ Mistral API Key 未設定，請更新 config.json"})

    client = Mistral(api_key=MISTRAL_API_KEY)
    context = json.dumps(retrieved_data, ensure_ascii=False, indent=4)
    prompt = f"用戶問題: {message}\n\n相關數據:\n{context}\n\n請根據以上資訊，並有效結合USITC Tariff Database Fields欄位對照、Tariff Database Code Key代碼意義對照表等資訊，給予完整的回答。在回答時，國家代碼需要轉換為國家名稱來進行清晰回應。如果有必要，可以提供額外資訊（例如：MFN關稅、日期等等）。除此之外，不要回答非關稅相關的問題。"


    messages = [{"role": "user", "content": prompt}]
    chat_response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        temperature=1
    )

    response_text = chat_response.choices[0].message.content
    history.append({"role": "assistant", "content": response_text})
    return history

with gr.Blocks() as demo:
    gr.Markdown("## 🗣 關稅查詢機器人")
    chat = gr.Chatbot(type="messages")
    user_input = gr.Textbox(placeholder="請輸入 HTS 編號或關鍵字...")
    send_button = gr.Button("送出")

    def send_message(message, history):
        if not message.strip():
            return "", history
        history.append({"role": "user", "content": message})
        return "", chatbot_response(message, history)

    send_button.click(send_message, [user_input, chat], [user_input, chat])

demo.launch(share=True)