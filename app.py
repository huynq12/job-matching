import certifi
from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
from pymongo import MongoClient
from pypdf import PdfReader
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import os
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import time
import math
from collections import Counter
import pandas as pd
from dotenv import load_dotenv
import json 
from bson.objectid import ObjectId

# Tải các resource cần thiết của NLTK
# try:
#     nltk.download('punkt_tab', quiet=True)
#     nltk.download('stopwords', quiet=True)

# except Exception as e:
#     print(f"NLTK download warning: {e}")
load_dotenv()
app = Flask(__name__)
CORS(app, resources={"/jobs/*": {"origins": "*"}})

# Cấu hình
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# Kết nối MongoDB
MONGO_URI_LOCAL = "mongodb://localhost:27017/"
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "job_matching"
JOBS_COLLECTION = "job_dataset"
JOBS_EMBEDDING = "job_embedding"
STOPWORDS_EN = "stopwords_en"
POS_TAG = "pos_tag"


def get_mongo_connection():
    # client = MongoClient(MONGO_URI,tlsCAFile=certifi.where())
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("Mongo URI: ", MONGO_URI)
    return client, db

client, db = get_mongo_connection()
stop_words_collection = db[STOPWORDS_EN]
job_collection = db[JOBS_COLLECTION]

# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('maxent_treebank_pos_tagger')
#     nltk.download('averaged_perceptron_tagger_eng', quiet=True)
# except Exception as e:
#     print(f"NLTK download warning: {e}")


def save_stop_words():
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
    
def extract_text_from_docx(file_path):
    text = ""
    try:
        document = Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Lỗi trích xuất text từ DOCX: {e}")
        return ""
    
def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        print(f"Định dạng file không được hỗ trợ: {file_extension}")
        return ""

def preprocess_text(text):
    try:
        return preprocess_text_v2(text)
        # text = text.lower()
        
        # text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # tokens = word_tokenize(text)

        # stop_words_docs = list(stop_words_collection.find({}, {'text': 1, '_id': 0}))
        # stop_words = set(doc['text'] for doc in stop_words_docs)

        # filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # return ' '.join(filtered_tokens)
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
   

def calculate_tfidf_docs(documents):
    idf_values = {}
    total_docs = len(documents)
    
    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            if word not in idf_values:
                idf_values[word] = 0
            idf_values[word] += 1
    
    for word in idf_values:
        idf_values[word] = math.log(total_docs / idf_values[word])
    
    tfidf_documents = []
    for doc in documents:
        tf_values = {}
        word_count = Counter(doc)
        doc_length = len(doc)
        for word, count in word_count.items():
            tf_values[word] = count / doc_length
        
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
    all_jobs = list(job_collection.find({}, {'_id': 1, 'job_description': 1, 'position_title': 1, 'model_response': 1, 'company': 1}))
    
    if not all_jobs:
        client.close()
        return None, None, []
    
    job_texts = [preprocess_text(job.get('job_description', '') + job.get('position_title','') + job.get('model_response', '')) for job in all_jobs]
    job_words = [[word for word in text.split()] for text in job_texts]
    
    tfidf_documents = calculate_tfidf_docs(job_words)
    
    # client.close()
    return all_jobs, tfidf_documents


def find_matching_jobs(resume_text, k=5):
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
    # print(neighbors)
    # get matching job best
    matching_jobs = []
    for idx, distance in neighbors:
        if idx < len(all_jobs):
            job = all_jobs[idx]
            # print(job)
            job_id = str(job.get('_id', 'unknown'))
            position_title = job.get('position_title')
            company = job.get('company')
            # print(job.get('company'))
            # print(job.get('position_title'))
            similarity_score = float(1 - distance)
            # similarity_score = distance
            # job_description = job.get('job_description', '')
            benefit = None
            model_response_str = job.get('model_response')
            if(model_response_str):
                try:
                    model_response_dict = json.loads(job['model_response'])
                    benefit = model_response_dict.get('Compensation and Benefits')
                except (TypeError, json.JSONDecodeError):
                    benefit = None
            
            matching_jobs.append({
                'id': job_id,
                'position_title': position_title,
                'company': company,
                'benefit': "Negotiable" if benefit == "N/A" else benefit,
                'similarity_score': similarity_score,
                # 'job_description': job_description
            })
    
    return matching_jobs

@app.route('/jobs/get-all', methods=['GET'])
def get_all_jobs():
    filter = request.args.get('filter', '')
    skip = request.args.get('skip', default = 0, type = int)
    take = request.args.get('take', default = 10, type = int)

    query = {}
    if filter:
        if ObjectId.is_valid(filter):
            query['_id'] = ObjectId(filter)
        else:
            query['$or'] = [
                {'company': {'$regex': filter, '$options': 'i'}},
                {'position_title': {'$regex': filter, '$options': 'i'}},
                # {'model_response':{'$regex': filter, '$options': 'i'}}
            ]

    total_count = job_collection.count_documents(query)

    jobs_cursor = job_collection.find(query, {'_id': 1, 'company': 1, 'position_title': 1, 'model_response': 1}).skip(skip).limit(take)
    items = list(jobs_cursor)

    custom_items = []

    for job in items:
        job_id = str(job['_id'])
        company = job.get('company')
        # print(job.get('company'))
        position_title = job.get('position_title')
        benefit = None
        model_response_str = job.get('model_response')
        if(model_response_str):
            try:
                model_response_dict = json.loads(job['model_response'])
                benefit = model_response_dict.get('Compensation and Benefits')
            except (TypeError, json.JSONDecodeError):
                benefit = None
        
        custom_job = {
            'id': job_id,
            'company': company,
            'position_title': position_title,
            'benefit': "Negotiable" if benefit == "N/A" else benefit
        }

        custom_items.append(custom_job)
        
    result = {
        'totalCount': total_count,
        'items': custom_items
    }

    response = {
        'isSuccess': True,
        'errorCode': None,
        'data': result,
        'message': None
    }

    return jsonify(response)

@app.route('/jobs/match-resume', methods=['POST'])
def match_resume():
    """API endpoint để match CV với các jobs trong MongoDB"""
    start_time = time.time()
    
    # Kiểm tra request có đủ file không
    if 'resume' not in request.files:
        return jsonify({"error": "Missing resume file"}), 400
    
    resume_file = request.files['resume']
    
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
        resume_text = extract_text_from_file(resume_path)
        if not resume_text:
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        # Tìm jobs phù hợp
        matching_jobs = find_matching_jobs(resume_text, k)
        
        # Xóa file tạm sau khi xử lý
        os.remove(resume_path)
        
        # processing_time = time.time() - start_time
        
        return jsonify({
            "isSuccess": True,
            "data": matching_jobs,
            "errorCode": None,
            "message": None,
        })
        
    except Exception as e:
        # Xóa file tạm nếu có lỗi
        if os.path.exists(resume_path):
            os.remove(resume_path)
        
        return jsonify({
            "isSuccess": False,
            "errorCode": str(e),
            "message": "An error occurred during processing",
            "data": None
        }), 500

@app.route('/save-stop-words', methods= ['POST'])
def save_stop_words_into_db():
    save_stop_words()
    return jsonify({
            "success": True,
        })

@app.route('/jobs/save-job-embedding', methods = ['GET'])
def save_job_embedding():
    client1 = MongoClient(MONGO_URI,tlsCAFile=certifi.where())
    db1 = client1[DB_NAME]
    job_embedding1 = db1["job-bedding"]
    data = list(job_embedding1.find({}, {'company': 1, 'position_title': 1, 'model_response': 1, 'embedding': 1}))

    client2 = MongoClient(MONGO_URI_LOCAL)
    db2 = client2[DB_NAME]
    job_embedding2 = db2[JOBS_EMBEDDING]
    if data:
        job_embedding2.insert_many(data)
        return jsonify({"success": True,
                        "message": f"{len(data)} documents copied successfully"})
    else:
        return jsonify({"success": True, "message": "No documents to copy"})

    
@app.route('/import-jobs', methods=['POST'])
def import_jobs():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)
    
    try:
        # Đọc dữ liệu từ CSV
        df = pd.read_csv(file_path)
        
        # Kiểm tra nếu cột thứ 3 tồn tại (job_description)
        if len(df.columns) < 3:
            return jsonify({"error": "CSV must have at least 3 columns"}), 400
        
        jobs_data = []
        for _, row in df.iterrows():
            job_entry = {
                "company": row.iloc[0],
                "job_description": row.iloc[1], 
                "position_title": row.iloc[2],
                "model_response": row.iloc[4]
            }
            jobs_data.append(job_entry)
        
        # Thêm dữ liệu vào MongoDB
        job_collection.insert_many(jobs_data)
        
        return jsonify({"success": True, "message": f"{len(jobs_data)} jobs imported successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Xóa file tạm sau khi xử lý
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint để kiểm tra trạng thái API"""
    try:
        # Kiểm tra kết nối MongoDB
        client, db = get_mongo_connection()
        job_count = db[JOBS_COLLECTION].count_documents({})
        client.close()
        
        # Kiểm tra model
        # model_status = "Not found"
        # if os.path.exists('tfidf_knn_model.pkl'):
        #     model_status = "Available"
        
        return jsonify({
            "status": "OK",
            "mongodb_connection": "Connected",
            "job_count": job_count,
            # "model_status": model_status
        })
    except Exception as e:
        return jsonify({
            "status": "Error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)