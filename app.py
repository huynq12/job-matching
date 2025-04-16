from flask import Flask, request, jsonify
from pymongo import MongoClient
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
import os
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import time
import math
from collections import Counter

# Tải các resource cần thiết của NLTK
# try:
#     nltk.download('punkt_tab', quiet=True)
#     nltk.download('stopwords', quiet=True)

# except Exception as e:
#     print(f"NLTK download warning: {e}")

app = Flask(__name__)

# Cấu hình
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# Kết nối MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "job_matching"
JOBS_COLLECTION = "job_dataset"
STOPWORDS_EN = "stopwords_en"
POS_TAG = "pos_tag"


def get_mongo_connection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return client, db

client, db = get_mongo_connection()
stop_words_collection = db[STOPWORDS_EN]

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('maxent_treebank_pos_tagger')
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    print(f"NLTK download warning: {e}")


def save_stop_words():
    client, db = get_mongo_connection()
    stop_words_collection = db[STOPWORDS_EN]

    nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))
    
    print(stop_words)

    stopwords_input = []
    for item in stop_words:
        entry = {
            "text": item,
        }
        stopwords_input.append(entry)
    
    # Thêm dữ liệu vào MongoDB
    stop_words_collection.insert_many(stopwords_input)
    return

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text(text):
    try:
        text = text.lower()
        
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        tokens = word_tokenize(text)

        stop_words_docs = list(stop_words_collection.find({}, {'text': 1, '_id': 0}))
        stop_words = set(doc['text'] for doc in stop_words_docs)

        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(filtered_tokens)
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        return text  

def preprocess_text_v2(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        sentences = sent_tokenize(text)

        stop_words_docs = list(stop_words_collection.find({}, {'text': 1, '_id': 0}))
        stop_words = set(doc['text'] for doc in stop_words_docs)

        processed_sentences = []

        for sent in sentences:
            # if any(criteria in sent for criteria in ['skills', 'education']):
                words = word_tokenize(sent)
                words = [word for word in words if word not in stop_words]
                tagged_words = pos_tag(words)
                filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
                processed_sentences.append(" ".join(filtered_words))

        return " ".join(processed_sentences)
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        return text  
   

# Hàm tính TF-IDF
def calculate_tfidf_docs(documents):
    # Tính toán IDF
    idf_values = {}
    total_docs = len(documents)
    
    # Đếm số tài liệu chứa mỗi từ
    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            if word not in idf_values:
                idf_values[word] = 0
            idf_values[word] += 1
    
    # Tính IDF cho mỗi từ
    for word in idf_values:
        idf_values[word] = math.log(total_docs / idf_values[word])
    
    # Tính toán TF-IDF cho từng tài liệu
    tfidf_documents = []
    for doc in documents:
        # Tính TF
        tf_values = {}
        word_count = Counter(doc)
        doc_length = len(doc)
        for word, count in word_count.items():
            tf_values[word] = count / doc_length
        
        # Tính TF-IDF
        tfidf_doc = {}
        for word in tf_values:
            tfidf_doc[word] = tf_values[word] * idf_values.get(word, 0)
        tfidf_documents.append(tfidf_doc)
    
    return tfidf_documents

def cosine_distance(vector1, vector2):
    common_keys = set(vector1.keys()) & set(vector2.keys())
    dot_product = sum(vector1[key] * vector2[key] for key in common_keys)
    magnitude1 = math.sqrt(sum(vector1[key]**2 for key in vector1))
    magnitude2 = math.sqrt(sum(vector2[key]**2 for key in vector2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 1  # zero vector
    return 1 - dot_product / (magnitude1 * magnitude2)

def find_knn(query_vector, vectors, k):
    distances = []
    for i, vector in enumerate(vectors):
        distance = cosine_distance(query_vector, vector)
        distances.append((i, distance))
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def build_tfidf_model():

    client, db = get_mongo_connection()
    jobs_collection = db[JOBS_COLLECTION]
    
    all_jobs = list(jobs_collection.find({}, {'_id': 1, 'job_description': 1, 'position_title': 1}))
    
    if not all_jobs:
        client.close()
        return None, None, []
    
    job_texts = [preprocess_text_v2(job.get('job_description', '') + job.get('position_title','')) for job in all_jobs]
    job_words = [[word for word in text.split()] for text in job_texts]
    
    tfidf_documents = calculate_tfidf_docs(job_words)
    
    client.close()
    return all_jobs, tfidf_documents


def find_matching_jobs(resume_text, k=5):
    # Tiền xử lý resume text
    processed_resume = preprocess_text(resume_text)
    resume_words = processed_resume.split()
    
    # Tải hoặc xây dựng mô hình TF-IDF và KNN
    model_path = 'tfidf_knn_model.pkl'
    tfidf_documents = None
    all_jobs = None
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                if 'tfidf_documents' in model_data and 'all_jobs' in model_data:
                    tfidf_documents = model_data['tfidf_documents']
                    all_jobs = model_data['all_jobs']
                else:
                    print("Missing required keys in the model file")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # If can not load, build new model
    if tfidf_documents is None or all_jobs is None:
        print("Building new TF-IDF model...")
        all_jobs, tfidf_documents = build_tfidf_model()
        # save model
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({'tfidf_documents': tfidf_documents, 'all_jobs': all_jobs}, f)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    if not tfidf_documents or not all_jobs:
        print("Failed to create or load TF-IDF model")
        return []
    
    # calc TF-IDF of resume
    # TF
    tf_values = {}
    for word in resume_words:
        if word not in tf_values:
            tf_values[word] = 0
        tf_values[word] += 1
    
    for word in tf_values:
        tf_values[word] /= len(resume_words)
    
    resume_tfidf = {}
    for word in tf_values:
        # only use TF if not idf value
        resume_tfidf[word] = tf_values[word]
        # resume_tfidf[word] = tf_values[word] * idf_values.get(word, 0)
    
    # find k nearest neighbors
    neighbors = find_knn(resume_tfidf, tfidf_documents, k)
    
    # get matching job best
    matching_jobs = []
    for idx, distance in neighbors:
        if idx < len(all_jobs):
            job = all_jobs[idx]
            job_id = str(job.get('_id', 'unknown'))
            position_title = job.get('position_title', 'Unknown')
            similarity_score = float(1 - distance)
            job_description = job.get('job_description', '')
            
            matching_jobs.append({
                'job_id': job_id,
                'position_title': position_title,
                'similarity_score': similarity_score,
                # 'job_description': job_description
            })
    
    return matching_jobs

# def build_tfidf_model():
#     """Xây dựng mô hình TF-IDF từ job descriptions trong MongoDB"""
#     client, db = get_mongo_connection()
#     jobs_collection = db[JOBS_COLLECTION]
    
#     # Lấy tất cả job descriptions từ MongoDB
#     all_jobs = list(jobs_collection.find({}, {'_id': 1, 'job_description': 1, 'position_title': 1}))
    
#     if not all_jobs:
#         client.close()
#         return None, None, []
    
#     # Tiền xử lý job descriptions
#     job_texts = [preprocess_text(job.get('job_description', '')) for job in all_jobs]
    
#     # Xây dựng TF-IDF vectorizer
#     vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
#     job_vectors = vectorizer.fit_transform(job_texts)
    
#     # Xây dựng KNN model
#     knn_model = NearestNeighbors(n_neighbors=min(10, len(all_jobs)), algorithm='auto', metric='cosine')
#     knn_model.fit(job_vectors)
    
#     client.close()
#     return vectorizer, knn_model, all_jobs

# def find_matching_jobs(resume_text, k=5):
#     """Tìm k job descriptions phù hợp nhất với resume"""
#     # Tiền xử lý resume text
#     processed_resume = preprocess_text(resume_text)
    
#     # Tải hoặc xây dựng mô hình TF-IDF và KNN
#     model_path = 'tfidf_knn_model.pkl'
#     if os.path.exists(model_path):
#         try:
#             with open(model_path, 'rb') as f:
#                 model_data = pickle.load(f)
#                 vectorizer = model_data['vectorizer']
#                 knn_model = model_data['knn_model']
#                 # Lấy lại thông tin job từ MongoDB để đảm bảo dữ liệu mới nhất
#                 client, db = get_mongo_connection()
#                 all_jobs = list(db[JOBS_COLLECTION].find({}, {'_id': 1, 'job_description': 1, 'position_title': 1}))
#                 client.close()
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             vectorizer, knn_model, all_jobs = build_tfidf_model()
#     else:
#         vectorizer, knn_model, all_jobs = build_tfidf_model()
#         # Lưu model để sử dụng sau này
#         try:
#             with open(model_path, 'wb') as f:
#                 pickle.dump({'vectorizer': vectorizer, 'knn_model': knn_model}, f)
#         except Exception as e:
#             print(f"Error saving model: {e}")
    
#     if not vectorizer or not knn_model or not all_jobs:
#         return []
    
#     # Chuyển đổi resume thành vector TF-IDF
#     resume_vector = vectorizer.transform([processed_resume])
    
#     # Tìm k neighbors gần nhất
#     distances, indices = knn_model.kneighbors(resume_vector, 
#                                               n_neighbors=min(k, len(all_jobs)))
    
#     # Lấy thông tin của các jobs phù hợp nhất
#     matching_jobs = []
#     for i, idx in enumerate(indices[0]):
#         if idx < len(all_jobs):
#             matching_jobs.append({
#                 'job_id': str(all_jobs[idx]['_id']),
#                 'position_title': all_jobs[idx].get('position_title', 'Unknown'),
#                 'similarity_score': float(1 - distances[0][i]),  # Chuyển khoảng cách thành độ tương đồng
#                 'job_description': all_jobs[idx].get('job_description', '')
#             })
    
#     return matching_jobs

@app.route('/save-stop-words', methods= ['POST'])
def save_stop_words_into_db():
    save_stop_words()
    return jsonify({
            "success": True,
        })

@app.route('/match-resume', methods=['POST'])
def match_resume():
    """API endpoint để match CV với các jobs trong MongoDB"""
    start_time = time.time()
    
    # Kiểm tra request có đủ file không
    if 'resume' not in request.files:
        return jsonify({"error": "Missing resume file"}), 400
    
    resume_file = request.files['resume']
    
    # Kiểm tra định dạng file
    if resume_file.filename == '' or not resume_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Resume must be a PDF file"}), 400
    
    # Số lượng jobs muốn matching
    k = request.form.get('k', 5)
    try:
        k = int(k)
    except:
        k = 5
    
    # Lưu file tạm thời
    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume_file.filename))
    
    try:
        resume_file.save(resume_path)
        
        # Trích xuất text từ PDF
        resume_text = extract_text_from_pdf(resume_path)
        if not resume_text:
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        # Tìm jobs phù hợp
        matching_jobs = find_matching_jobs(resume_text, k)
        
        # Xóa file tạm sau khi xử lý
        os.remove(resume_path)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "matched_jobs": matching_jobs,
            "processing_time": processing_time
        })
        
    except Exception as e:
        # Xóa file tạm nếu có lỗi
        if os.path.exists(resume_path):
            os.remove(resume_path)
        
        return jsonify({
            "error": str(e),
            "details": "An error occurred during processing"
        }), 500

@app.route('/rebuild-model', methods=['GET'])
def rebuild_model():
    """API endpoint để xây dựng lại mô hình TF-IDF và KNN"""
    try:
        start_time = time.time()
        vectorizer, knn_model, all_jobs = build_tfidf_model()
        
        if not vectorizer or not knn_model:
            return jsonify({"error": "Failed to build model - no job data found"}), 400
        
        # Lưu model
        with open('tfidf_knn_model.pkl', 'wb') as f:
            pickle.dump({'vectorizer': vectorizer, 'knn_model': knn_model}, f)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "job_count": len(all_jobs),
            "processing_time": processing_time,
            "message": "Model rebuilt successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint để kiểm tra trạng thái API"""
    try:
        # Kiểm tra kết nối MongoDB
        client, db = get_mongo_connection()
        job_count = db[JOBS_COLLECTION].count_documents({})
        client.close()
        
        # Kiểm tra model
        model_status = "Not found"
        if os.path.exists('tfidf_knn_model.pkl'):
            model_status = "Available"
        
        return jsonify({
            "status": "OK",
            "mongodb_connection": "Connected",
            "job_count": job_count,
            "model_status": model_status
        })
    except Exception as e:
        return jsonify({
            "status": "Error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)