import pandas as pd
import json
import os
import fitz  # PyMuPDF 讀取 PDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # 使用本地 HuggingFace 嵌入
from rank_bm25 import BM25Okapi
from mistralai import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, request, abort, jsonify
from google.cloud import storage
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import time
import traceback

app = Flask(__name__)

# ✅ 環境變數
GCS_BUCKET_NAME = "usa_tax_2"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = None
handler = None

# ✅ 全域資料結構
embedding_model = None
vector_store = None
bm25 = None
combined_data = []

# ✅ 檢查環境變數
def check_env():
    required_vars = {
        "GCS_BUCKET_NAME": GCS_BUCKET_NAME,
        "MISTRAL_API_KEY": MISTRAL_API_KEY,
        "LINE_CHANNEL_ACCESS_TOKEN": LINE_CHANNEL_ACCESS_TOKEN,
        "LINE_CHANNEL_SECRET": LINE_CHANNEL_SECRET
    }
    
    missing_vars = [name for name, value in required_vars.items() if not value]
    
    if missing_vars:
        print(f"❌ 缺少必要的環境變數: {', '.join(missing_vars)}")
        print("❌ 請檢查 Cloud Function 設定")
        os._exit(1)
    
    print("✅ 環境變數檢查通過")

# ✅ 初始化 LINE BOT
def setup_linebot():
    global line_bot_api, handler
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
    print("✅ LINE BOT 已初始化")

# ✅ 延遲註冊 LINE handler
def register_handler():
    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        user_msg = event.message.text
        reply = generate_reply(user_msg)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

# ✅ 從 GCS 讀取 JSON 檔案
def load_json_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = blob.download_as_text()
    return json.loads(content)

# ✅ GCS 讀取 PDF
def load_pdfs_from_gcs(bucket_name, pdf_files):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    pdf_texts = []
    for pdf_file in pdf_files:
        blob = bucket.blob(pdf_file)
        content = blob.download_as_bytes()
        with fitz.open(stream=content, filetype="pdf") as doc:
            text = "\n".join([page.get_text("text") for page in doc])
            pdf_texts.append(text)
    return pdf_texts

# ✅ 將 FAISS 向量庫保存到 GCS
def save_vector_store_to_gcs(vector_store, bucket_name, directory_path="vector_store"):
    """將 FAISS 向量庫保存到 GCS"""
    import tempfile
    import shutil
    import os
    
    # 首先保存到臨時目錄
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save_local(temp_dir)
        
        # 上傳到 GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # 上傳 index.faiss 文件
        index_blob = bucket.blob(f"{directory_path}/index.faiss")
        index_blob.upload_from_filename(os.path.join(temp_dir, "index.faiss"))
        
        # 上傳 index.pkl 文件
        pkl_blob = bucket.blob(f"{directory_path}/index.pkl")
        pkl_blob.upload_from_filename(os.path.join(temp_dir, "index.pkl"))
    
    print(f"✅ 成功將向量庫保存到 GCS: gs://{bucket_name}/{directory_path}/")
    return True

# ✅ 從 GCS 加載 FAISS 向量庫 (改進版)
def load_vector_store_from_gcs(embedding_model, bucket_name, directory_path="vector_store"):
    """從 GCS 加載 FAISS 向量庫，增強兼容性和錯誤處理"""
    import tempfile
    import os
    from langchain_community.vectorstores import FAISS
    
    try:
        # 創建臨時目錄
        temp_dir = tempfile.mkdtemp()
        
        # 從 GCS 下載文件
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # 下載 index.faiss 文件
        index_blob = bucket.blob(f"{directory_path}/index.faiss")
        index_file_path = os.path.join(temp_dir, "index.faiss")
        index_blob.download_to_filename(index_file_path)
        
        # 下載 index.pkl 文件
        pkl_blob = bucket.blob(f"{directory_path}/index.pkl")
        pkl_file_path = os.path.join(temp_dir, "index.pkl")
        pkl_blob.download_to_filename(pkl_file_path)
        
        print(f"✅ 已從 GCS 下載向量庫文件: gs://{bucket_name}/{directory_path}/")
        
        # 輸出文件大小信息，有助於診斷
        faiss_size = os.path.getsize(index_file_path) / (1024 * 1024)
        pkl_size = os.path.getsize(pkl_file_path) / (1024 * 1024)
        print(f"📊 向量庫文件大小: index.faiss: {faiss_size:.2f}MB, index.pkl: {pkl_size:.2f}MB")
        
        # 添加診斷代碼：檢查當前嵌入模型的向量維度
        current_embedding_dim = len(embedding_model.embed_query("診斷向量維度"))
        print(f"🔍 當前嵌入模型向量維度: {current_embedding_dim}")
        
        # 嘗試加載向量庫
        try:
            vector_store = FAISS.load_local(temp_dir, embedding_model, allow_dangerous_deserialization=True)
            
            # 輸出一些向量庫信息
            if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'ntotal'):
                print(f"📊 加載的向量庫包含 {vector_store.index.ntotal} 個向量")
                
                # 檢查向量庫中向量的維度
                if hasattr(vector_store.index, 'd'):
                    print(f"🔍 向量庫中向量的維度: {vector_store.index.d}")
                    
                    # 如果維度不匹配，給出警告
                    if vector_store.index.d != current_embedding_dim:
                        print(f"⚠️ 警告：向量庫維度 ({vector_store.index.d}) 與當前模型維度 ({current_embedding_dim}) 不匹配")
                        print("建議重新建立向量庫")
            
            print(f"✅ 成功從 GCS 加載向量庫")
            return vector_store
        
        except Exception as e:
            print(f"❌ 加載向量庫文件失敗: {str(e)}")
            print("⚠️ 可能是向量維度或格式與當前嵌入模型不兼容")
            traceback.print_exc()
            return None
    
    except Exception as e:
        print(f"❌ 從 GCS 下載向量庫文件時出錯: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # 清理臨時目錄
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)


# ✅ 檢查 GCS 中是否存在向量庫
def vector_store_exists_in_gcs(bucket_name, directory_path="vector_store"):
    """檢查 GCS 中是否存在向量庫"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # 檢查 index.faiss 文件是否存在
        index_blob = bucket.blob(f"{directory_path}/index.faiss")
        index_exists = index_blob.exists()
        
        # 檢查 index.pkl 文件是否存在
        pkl_blob = bucket.blob(f"{directory_path}/index.pkl")
        pkl_exists = pkl_blob.exists()
        
        return index_exists and pkl_exists
    except Exception as e:
        print(f"❌ 檢查 GCS 向量庫時出錯: {str(e)}")
        return False

# ✅ 安全處理嵌入批次
def safe_embed_documents(embedding_model, texts, batch_size=8, max_texts=8000):
    """安全地處理文本嵌入，支援批次處理，並限制最大文本數量"""
    
    # 限制處理的文本數量
    #if len(texts) > max_texts:
    #    print(f"⚠️ 文本數量過多 ({len(texts)})，限制為 {max_texts} 個")
    #    texts = texts[:max_texts]
    
    # 將文本分成較小的批次
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    all_embeddings = []
    valid_texts = []
    
    for i, batch in enumerate(batches):
        print(f"▶️ 處理批次 {i+1}/{len(batches)}，共 {len(batch)} 個文本")
        
        try:
            # 嘗試批量嵌入
            embeddings = embedding_model.embed_documents(batch)
            
            # 檢查嵌入結果是否有效
            if len(embeddings) == len(batch):
                print(f"✅ 批次 {i+1} 嵌入成功，獲得 {len(embeddings)} 個向量")
                all_embeddings.extend(embeddings)
                valid_texts.extend(batch)
            else:
                print(f"⚠️ 批次 {i+1} 嵌入結果數量不符，預期 {len(batch)}，實際 {len(embeddings)}")
                # 只使用成功的部分
                all_embeddings.extend(embeddings)
                valid_texts.extend(batch[:len(embeddings)])
        except Exception as e:
            print(f"❌ 批次 {i+1} 嵌入處理失敗: {str(e)}")
            traceback.print_exc()
        
        # 批次之間添加短暫間隔，減輕計算負擔
        time.sleep(0.1)
    
    print(f"📊 總計: 處理 {len(texts)} 個文本，成功嵌入 {len(all_embeddings)} 個")
    return valid_texts, all_embeddings

# ✅ Cold start / Init 用：初始化資料與模型
def do_init():
    global combined_data, embedding_model, vector_store, bm25

    JSON_FILE = "trade_tariff_database_2025.json"
    PDF_FILES = ["td-codes.pdf", "finalCopy.pdf", "td-fields.pdf"]
    VECTOR_STORE_DIR = "vector_store"  # GCS 中存儲向量庫的目錄

    try:
        print("▶️ 從 GCS 載入 JSON 與 PDF...")
        database = load_json_from_gcs(GCS_BUCKET_NAME, JSON_FILE)
        pdf_data = load_pdfs_from_gcs(GCS_BUCKET_NAME, PDF_FILES)
        combined_data = database + [{"content": text} for text in pdf_data]
    except Exception as e:
        print(f"❌ 初始化資料載入失敗：{e}")
        traceback.print_exc()
        return False

    if not combined_data:
        print("❌ 未讀取到任何有效數據")
        return False

    print("▶️ 初始化 HuggingFace 本地嵌入模型...")
    try:
        # 使用本地 HuggingFaceEmbeddings 模型
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"  
        )
        
        # 測試嵌入模型
        test_result = embedding_model.embed_query("測試嵌入模型")
        if isinstance(test_result, list) and len(test_result) > 0:
            print(f"✅ 嵌入模型測試成功，向量維度: {len(test_result)}")
        else:
            print(f"⚠️ 嵌入模型測試結果格式異常: {type(test_result)}")
            embedding_model = None
            
    except Exception as e:
        print(f"❌ 嵌入模型初始化失敗: {str(e)}")
        traceback.print_exc()
        embedding_model = None
    
    # 檢查是否存在預先建立的向量庫
    if embedding_model:
        print("▶️ 檢查 GCS 中是否存在向量庫文件...")
        if vector_store_exists_in_gcs(GCS_BUCKET_NAME, VECTOR_STORE_DIR):
            print("✅ 發現向量庫文件，正在嘗試加載...")
            
            # 記錄當前時間，用於計算加載時間
            start_time = time.time()
            
            vector_store = load_vector_store_from_gcs(embedding_model, GCS_BUCKET_NAME, VECTOR_STORE_DIR)
            
            if vector_store:
                load_time = time.time() - start_time
                print(f"✅ 成功加載向量庫，耗時 {load_time:.2f} 秒")
                
                # 嘗試檢查向量維度是否匹配
                test_embedding = embedding_model.embed_query("測試")
                expected_dim = len(test_embedding)
                
                # 嘗試從向量庫獲取一個向量來檢查維度
                try:
                    # 使用一個簡單查詢來檢查向量
                    test_results = vector_store.similarity_search("test", k=1)
                    if test_results:
                        print(f"✅ 向量庫可以正常搜索，載入完成")
                    else:
                        print("⚠️ 向量庫加載成功，但搜索測試返回空結果")
                except Exception as e:
                    print(f"⚠️ 向量庫加載後，搜索測試失敗: {str(e)}")
                    print("⚠️ 繼續使用向量庫，但可能存在兼容性問題")
            else:
                print("⚠️ 加載向量庫失敗，將重新建立")
                vector_store = None
        else:
            print("ℹ️ 未找到現有向量庫，將建立新的向量庫")
            vector_store = None
    
    # 如果沒有加載成功，則重新建立向量庫
    if embedding_model and not vector_store:
        # 準備文本數據
        text_data = [record.get("content", "") for record in combined_data]
        text_splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text("\n".join(text_data))
        print(f"▶️ 文本拆分完成，共 {len(text_splits)} 個片段")
        
        # 限制處理的文本數量，避免超時
        #max_texts = 8000  # 可以根據您的 Cloud Function 資源來調整
        #if len(text_splits) > max_texts:
        #    print(f"⚠️ 限制處理的文本數量為 {max_texts}（原有 {len(text_splits)} 個）")
        #    text_splits = text_splits[:max_texts]
        
        try:
            # 使用安全的方式處理嵌入
            valid_texts, embeddings = safe_embed_documents(embedding_model, text_splits, batch_size=8)
            
            if valid_texts and embeddings:
                # 建立向量庫
                text_embeddings = list(zip(valid_texts, embeddings))
                try:
                    vector_store = FAISS.from_embeddings(text_embeddings, embedding_model)
                    print(f"✅ 成功建立向量庫，包含 {len(text_embeddings)} 個嵌入")
                    
                    # 將向量庫保存到 GCS
                    print("▶️ 將向量庫保存到 GCS...")
                    if save_vector_store_to_gcs(vector_store, GCS_BUCKET_NAME, VECTOR_STORE_DIR):
                        print("✅ 向量庫已成功保存到 GCS，下次將直接加載")
                    else:
                        print("⚠️ 向量庫保存失敗，但本次仍可使用")
                        
                except Exception as e:
                    print(f"❌ 建立向量庫失敗: {str(e)}")
                    traceback.print_exc()
                    vector_store = None
            else:
                print("❌ 沒有有效的嵌入結果，無法建立向量庫")
                vector_store = None
        except Exception as e:
            print(f"❌ 嵌入處理失敗: {str(e)}")
            traceback.print_exc()
            vector_store = None
    else:
        if not embedding_model:
            print("⚠️ 嵌入模型未初始化，跳過向量庫建立/加載")
    
    # 建立 BM25 索引（作為備用搜索方法）
    print("▶️ 建立 BM25 關鍵字搜尋索引...")
    try:
        bm25_corpus = [record.get("content", "") for record in combined_data if "content" in record]
        if bm25_corpus:
            tokenized_corpus = [doc.split() for doc in bm25_corpus if doc.strip()]
            if tokenized_corpus:
                bm25 = BM25Okapi(tokenized_corpus)
                print(f"✅ 成功建立 BM25 索引，包含 {len(tokenized_corpus)} 個文檔")
            else:
                print("❌ 沒有有效的分詞文本，無法建立 BM25 索引")
                bm25 = None
        else:
            print("❌ 沒有有效的文本內容，無法建立 BM25 索引")
            bm25 = None
    except Exception as e:
        print(f"❌ 建立 BM25 索引失敗: {str(e)}")
        traceback.print_exc()
        bm25 = None
    
    # 如果所有索引都失敗，則返回失敗
    if not vector_store and not bm25:
        print("❌ 向量庫和 BM25 索引都初始化失敗，系統將無法正常工作")
        return False
    
    print("✅ 初始化完成")
    return True

# ✅ Webhook：LINE 訊息入口
@app.route("/callback", methods=['POST'])
def callback(request):
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    print(f"✅ 收到 LINE Webhook 請求")
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print(f"❌ LINE 簽名驗證失敗")
        abort(400)
    except Exception as e:
        print(f"❌ LINE Webhook 處理失敗: {str(e)}")
        traceback.print_exc()
        abort(500)
    
    return 'OK'

# ✅ 手動初始化 API
@app.route("/init", methods=['POST'])
def initialize():
    success = do_init()
    if success:
        return jsonify({"status": "success", "msg": "資料與模型初始化完成"})
    else:
        return jsonify({"status": "fail", "msg": "初始化失敗"}), 500

# ✅ 強制重建向量庫 API
@app.route("/rebuild-vectors", methods=['POST'])
def rebuild_vectors():
    """強制重建向量庫的 API 端點"""
    global vector_store, embedding_model
    
    # 檢查嵌入模型是否已初始化
    if not embedding_model:
        return jsonify({
            "status": "error", 
            "msg": "嵌入模型未初始化，請先呼叫 /init"
        }), 400
    
    try:
        # 準備文本數據
        text_data = [record.get("content", "") for record in combined_data]
        text_splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text("\n".join(text_data))
        
        # 限制處理的文本數量，避免超時
        #max_texts = 8000  # 可以根據您的 Cloud Function 資源來調整
        #if len(text_splits) > max_texts:
        #    text_splits = text_splits[:max_texts]
        
        # 使用安全的方式處理嵌入
        valid_texts, embeddings = safe_embed_documents(embedding_model, text_splits, batch_size=8)
        
        if not valid_texts or not embeddings:
            return jsonify({
                "status": "error", 
                "msg": "嵌入處理失敗，沒有獲取到有效的嵌入向量"
            }), 500
        
        # 建立向量庫
        text_embeddings = list(zip(valid_texts, embeddings))
        vector_store = FAISS.from_embeddings(text_embeddings, embedding_model)
        
        # 將向量庫保存到 GCS
        VECTOR_STORE_DIR = "vector_store"
        save_success = save_vector_store_to_gcs(vector_store, GCS_BUCKET_NAME, VECTOR_STORE_DIR)
        
        if not save_success:
            return jsonify({
                "status": "partial_success", 
                "msg": "向量庫已重建但保存失敗，重啟後將需要重新建立"
            })
        
        return jsonify({
            "status": "success", 
            "msg": f"成功重建並保存向量庫，包含 {len(text_embeddings)} 個嵌入向量"
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "msg": f"重建向量庫時發生錯誤: {str(e)}"
        }), 500

# ✅ 健康檢查 API
@app.route("/healthz", methods=['GET'])
def healthz():
    return "OK"

@app.route("/", methods=['GET'])
def index():
    return "Service is running"

# ✅ 檢索資料
def retrieve_relevant_records(query):
    keyword_matches, vector_matches = [], []
    
    # 使用 BM25 進行關鍵字搜尋
    if bm25:
        try:
            print(f"▶️ 執行 BM25 關鍵字搜尋...")
            tokenized_query = query.split()
            
            # 確保查詢不為空
            if not tokenized_query:
                tokenized_query = ["empty"]
            
            bm25_scores = bm25.get_scores(tokenized_query)
            top_indexes = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
            
            # 獲取對應的文檔
            for i in top_indexes:
                if i < len(combined_data):
                    keyword_matches.append(combined_data[i])
            
            print(f"✅ BM25 搜尋找到 {len(keyword_matches)} 個匹配項")
        except Exception as e:
            print(f"⚠️ BM25 搜尋失敗: {str(e)}")
            traceback.print_exc()
    
    # 使用向量庫進行語意搜尋
    if vector_store:
        try:
            print(f"▶️ 執行向量相似度搜尋...")
            faiss_results = vector_store.similarity_search(query, k=5)
            vector_matches = [{"content": doc.page_content} for doc in faiss_results]
            print(f"✅ 向量搜尋找到 {len(vector_matches)} 個匹配項")
        except Exception as e:
            print(f"⚠️ 向量搜尋失敗: {str(e)}")
            traceback.print_exc()
    
    # 合併搜尋結果並去重
    results = list({json.dumps(r, ensure_ascii=False) for r in (keyword_matches + vector_matches)})
    
    if not results:
        print("⚠️ 沒有找到任何匹配結果")
        return [{"message": "未找到相關資訊"}]
    
    return [json.loads(r) for r in results]

# ✅ 使用 Mistral API 生成回應
def generate_reply(message):
    # 檢查系統是否已初始化
    if not (embedding_model or bm25):
        return "⚠️ 系統尚未初始化，請先呼叫 /init"

    try:
        # 檢索相關記錄
        retrieved_data = retrieve_relevant_records(message)
        
        # 準備 Mistral API 請求
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        # 將檢索結果轉換為上下文
        context = json.dumps(retrieved_data, ensure_ascii=False, indent=2)
        
        # 構建提示詞
        prompt = f"""你是一個專業的美國國際貿易委員會（United States International Trade Commission / USITC）關稅數據專家助手。你的任務是從提供的數據中，準確且清晰地回答與關稅、貿易相關的問題。

                    處理問題時，請遵循以下指導原則：

                    1. 數據來源與可靠性
                    - 僅使用提供的數據庫內容作為回答依據
                    - 如果數據不足，明確說明無法完全回答

                    2. 回答格式要求
                    - 使用清晰的列點格式
                    - 避免使用複雜的符號
                    - 將專業術語轉換為易於理解的語言
                    - 優先提供最關鍵和最相關的信息

                    3. 回答必須包含
                    - 具體的關稅數據
                    - 國家代碼的對應國家名稱
                    - 相關的 MFN（最惠國）關稅信息
                    - 可能的生效日期

                    4. 額外訊息補充
                    - 如有相關的附加貿易背景信息，可以簡要補充
                    - 提供數據的來源和參考依據

                    5. 限制條件
                    - 嚴格限定在關稅和國際貿易領域
                    - 對於超出範圍的問題，禮貌地拒絕回答

                    用戶問題: {message}

                    相關數據:
                    {context}

                    請根據上述指導原則，提供一個專業、精確且易於理解的回答。"""

        # 發送請求到 Mistral API
        messages = [{"role": "user", "content": prompt}]
        
        try:
            chat_response = client.chat.complete(
                model="mistral-small-latest", 
                messages=messages, 
                temperature=0.3,
                max_tokens=300
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            print(f"❌ Mistral API 調用失敗: {str(e)}")
            traceback.print_exc()
            return f"⚠️ 無法生成回應，請稍後再試。技術錯誤: {str(e)[:100]}..."
    
    except Exception as e:
        print(f"❌ 生成回答時發生錯誤: {str(e)}")
        traceback.print_exc()
        return "⚠️ 處理您的請求時發生錯誤，請稍後再試。"

# ✅ 程式進入點：初始化系統
check_env()
setup_linebot()
print("📦 正在自動初始化系統資料...")
do_init()
register_handler()

# ✅ 本地測試用
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)