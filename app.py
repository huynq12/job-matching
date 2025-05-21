import certifi
from flask import Flask, request, jsonify, send_from_directory
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
import torch
from transformers import BertTokenizer, BertModel
import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from functools import wraps
from dataclasses import dataclass
from bson.objectid import ObjectId
# Tải các resource cần thiết của NLTK
# try:
#     nltk.download('punkt_tab', quiet=True)
#     nltk.download('stopwords', quiet=True)

# except Exception as e:
#     print(f"NLTK download warning: {e}")
load_dotenv()
app = Flask(__name__)
CORS(app, resources={"/*": {"origins": "*"}})

# Cấu hình
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# Kết nối MongoDB
MONGO_URI_LOCAL = "mongodb://localhost:27017/"
MONGO_URI = os.getenv("MONGO_URI")
MONGO_URI_ATLAS = os.getenv("MONGO_URI_ATLAS")
DB_NAME = "job_matching"
JOBS_COLLECTION = "job_dataset"
JOBS_EMBEDDING = "job_embedding"
STOPWORDS_EN = "stopwords_en"
POS_TAG = "pos_tag"
USERS_COLLECTION = "users"
JOB_BY_USER = "job_by_user"

def get_mongo_connection():
    # client = MongoClient(MONGO_URI,tlsCAFile=certifi.where())
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("Mongo URI: ", MONGO_URI)
    return client, db

client, db = get_mongo_connection()
stop_words_collection = db[STOPWORDS_EN]
job_collection = db[JOBS_COLLECTION]
job_embedding = db[JOBS_EMBEDDING]
users_collection = db[USERS_COLLECTION]
job_by_user = db[JOB_BY_USER]

#Authen
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Cấu hình password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Models (using dataclasses)
@dataclass
class UserCreate:
    username: str
    fullname: str
    email: str
    phone_number: str
    password: str
@dataclass
class UserResponse:
    id: str
    username: str
    created_at: datetime
    fullname: str
    email: str
    phone_number: str
@dataclass
class Token:
    access_token: str
    token_type: str = "bearer"

@dataclass
class TokenData:
    username: str = None

# Helper functions (same as before)
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    user = users_collection.find_one({"username": username})
    if user:
        user["_id"] = str(user["_id"])  # Convert ObjectId to string
        # print(user.username)
    return user

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Flask-specific helper for getting current user
def authorized(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"detail": "Not authenticated"}), 401

        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return jsonify({"detail": "Could not validate credentials"}), 401
            token_data = TokenData(username=username)
        except jwt.PyJWTError:
            return jsonify({"detail": "Could not validate credentials"}), 401
        user = get_user(token_data.username)
        if user is None:
            return jsonify({"detail": "Could not validate credentials"}), 401

        return f(current_user=user, *args, **kwargs)  # Inject user into the route

    return decorated_function

# Routes
@app.route("/register", methods=["POST"])
def register():
    user_data = request.get_json()
    user = UserCreate(**user_data)  # Create UserCreate object

    print(user)
    # Kiểm tra username đã tồn tại chưa
    db_user = get_user(user.username)
    if db_user:
        return jsonify({"detail": "Username already registered"}), 400

    # Tạo user mới với password đã hash
    hashed_password = get_password_hash(user.password)
    new_user_data = {
        "username": user.username,
        "password": hashed_password,
        "created_at": datetime.utcnow(),
        "fullname": user.fullname,
        "email": user.email,
        "phone_number": user.phone_number
    }

    # Lưu vào database
    result = users_collection.insert_one(new_user_data)

    # Trả về thông tin user (không bao gồm password)
    created_user = UserResponse(
        id=str(result.inserted_id),
        username=user.username,
        created_at=new_user_data["created_at"],
        fullname=user.fullname,
        email=user.email,
        phone_number=user.phone_number
    )
    print(created_user)

    response = {
        "isSuccess": True,
        "message": "User has been created successfully",
        "data": created_user.__dict__,
        "errorCode": None
    }

    return jsonify(response), 200 # Convert to dict for JSON

@app.route("/connect", methods=["POST"])
def login_for_access_token():
    form_data = request.get_json()
    username = form_data.get("username")
    password = form_data.get("password")

    user = authenticate_user(username, password)
    if not user:
        return jsonify({"detail": "Incorrect username or password"}), 401

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return jsonify(Token(access_token=access_token).__dict__), 200

@app.route("/user/profile", methods=["GET"])
@authorized
def read_users_me(current_user):
    user = UserResponse(
        id=current_user["_id"],
        username=current_user["username"],
        created_at=current_user["created_at"],
        fullname=current_user.get("fullname"),
        email=current_user.get("email"), 
        phone_number=current_user.get("phone_number")
    )
    return jsonify(user.__dict__), 200

# Test route
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Authentication API is running"}), 200
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('maxent_treebank_pos_tagger')
#     nltk.download('averaged_perceptron_tagger_eng', quiet=True)
# except Exception as e:
#     print(f"NLTK download warning: {e}")
@app.route("/user/job-by-user", methods=["GET"])
@authorized
def get_matched_jos_by_user(current_user):
    try:
        user_id = current_user["_id"]
        jobs = list(job_by_user.find({'user_id': user_id}, {'_id': 0,'job_id':1, 'position_title': 1, 'company': 1, 'benefit': 1, 'similarity_score': 1}))
        return jsonify({
                "isSuccess": False,
                "errorCode": None,
                "message": None,
                "data": jobs
            }), 200
    except Exception as e:
        return jsonify({
            "isSuccess": False,
            "errorCode": str(e),
            "message": "An error occurred during processing",
            "data": None
        }), 500
        

@app.route("/user/find-job-user", methods=["POST"])
@authorized
def find_matched_job(current_user):

    user_id = current_user["_id"]
    user_id_obj = ObjectId(user_id)


    if 'resume' not in request.files:
        return jsonify({"error": "Missing resume file"}), 400
    
    resume_file = request.files['resume']
    filename = secure_filename(resume_file.filename)
    resume_path_in_db = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Số lượng jobs muốn matching
    k = request.form.get('k', 5)
    try:
        k = int(k)
    except:
        k = 5
    
    # Lưu file tạm thời
    resume_path_on_disk = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        resume_file.save(resume_path_on_disk)
        
        
        # Trích xuất text từ PDF
        resume_text = extract_text_from_file(resume_path_on_disk)
        if not resume_text:
            os.remove(resume_path_on_disk)
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        # Tìm jobs phù hợp
        matching_jobs = find_matching_jobs_with_knn(resume_text, k)

        current_jobs = list(job_by_user.find({'user_id': user_id}))

        if(len(matching_jobs) > 0 and len(current_jobs) > 0):
            job_by_user.delete_many({'user_id': user_id})

        matching_jobs_with_user_id = []
        
        for job in matching_jobs:
            job['job_id'] = job.pop('id')
            job['user_id'] = user_id
            matching_jobs_with_user_id.append(job)

        job_by_user.insert_many(matching_jobs_with_user_id)

        print(resume_path_in_db)
        print(user_id)

        result = users_collection.update_one(
            {'_id': user_id_obj},
            {'$set': {'resume_path': resume_path_in_db, 'resume_filename': filename}}
        )
        print(f"Matched count: {result.matched_count}")
        print(f"Modified count: {result.modified_count}")

        # Xóa file tạm sau khi xử lý
        # os.remove(resume_path_on_disk)

        return jsonify({
            "isSuccess": False,
            "errorCode": None,
            "message": None,
            "data": None
        }), 200

    except Exception as e:
        # Xóa file tạm nếu có lỗi
        if os.path.exists(resume_path_on_disk):
            os.remove(resume_path_on_disk)
        
        return jsonify({
            "isSuccess": False,
            "errorCode": str(e),
            "message": "An error occurred during processing",
            "data": None
        }), 500


@app.route('/user/export-resume', methods=['GET'])
@authorized
def download_resume(current_user):
    user_id = current_user['_id']
    user_id_obj = ObjectId(user_id)

    user = users_collection.find_one({'_id': user_id_obj})
    if user and 'resume_path' in user:
        resume_path = user['resume_path']
        filename = user.get('resume_filename', 'resume.pdf') # Lấy tên file hoặc đặt mặc định
        return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.basename(resume_path), as_attachment=True, download_name=filename)
    else:
        return jsonify({"error": "Resume not found for this user"}), 404
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
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        sentences = sent_tokenize(text)

        stop_words_docs = list(stop_words_collection.find({}, {'text': 1, '_id': 0}))
        stop_words = set(doc['text'] for doc in stop_words_docs)

        processed_sentences = []

        for sent in sentences:
            if any(criteria in sent for criteria in ['skills', 'education']):
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
    all_jobs = list(job_embedding.find({}, {'_id': 1, 'job_description': 1, 'position_title': 1, 'model_response': 1, 'company': 1}))
    
    if not all_jobs:
        client.close()
        return None, None, []
    
    job_texts = [preprocess_text(job.get('job_description', '') + job.get('position_title','') + job.get('model_response', '')) for job in all_jobs]
    job_words = [[word for word in text.split()] for text in job_texts]
    
    tfidf_documents = calculate_tfidf_docs(job_words)
    
    # client.close()
    return all_jobs, tfidf_documents

def find_matching_jobs_with_knn(resume_text, k=5):
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
            job_id = str(job.get('_id', 'unknown'))
            position_title = job.get('position_title')
            company = job.get('company')
            similarity_score = float(1 - distance)
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
            })
    
    return matching_jobs

  
@app.route('/jobs/get-all', methods=['GET'])
# @authorized
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

@app.route('/jobs/match-resume-knn', methods=['POST'])
def match_resume_knn():
    """API endpoint để match CV với các jobs trong MongoDB"""
    # start_time = time.time()
    
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
        matching_jobs = find_matching_jobs_with_knn(resume_text, k)
        
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


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
def get_bert_embedding(text):
    try:
        # Cắt text nếu quá dài (BERT có giới hạn 512 tokens)
        max_length = 512
        
        # Tokenize và encode
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding='max_length')
        
        # Không tính gradient vì chỉ cần forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Lấy vector embedding của token [CLS] (đại diện cho cả câu)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        
        return embeddings.tolist()
    except Exception as e:
        # logger.error(f"Error generating BERT embedding: {e}")
        return None

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route('/jobs/match-resume-bert', methods=['POST'])
def find_matching_jobs_bert():
    if 'resume' not in request.files:
        return jsonify({"error": "No CV file uploaded"}), 400
    
    resume_file = request.files['resume']
    if resume_file.filename == '':
        return jsonify({"error": "No selected CV file"}), 400
    
    top_n = request.args.get('k', default = 5, type=int)

    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume_file.filename))
    
    try:
        resume_file.save(resume_path)
        
        # Trích xuất text từ PDF
        resume_text = extract_text_from_file(resume_path)
        if not resume_text:
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        # Tính embedding cho CV
        cv_embedding = get_bert_embedding(preprocess_text(resume_text))
        # print(cv_embedding)
        
        if cv_embedding is None:
            return jsonify({"error": "Could not generate embedding for CV"}), 500
        
        # Lấy tất cả job từ MongoDB
        jobs = list(job_embedding.find({}, {'_id': 1, 'company': 1, 'position_title': 1, 'job_description': 1, 'model_response': 1, 'embedding': 1}))
        
        # Tính cosine similarity giữa CV và mỗi job
        matched_jobs = []
        for job in jobs:
            # Kiểm tra job có embedding không
            if 'embedding' not in job:
                print(job)
            benefit = None
            model_response_str = job.get('model_response')
            if(model_response_str):
                try:
                    model_response_dict = json.loads(job['model_response'])
                    benefit = model_response_dict.get('Compensation and Benefits')
                except (TypeError, json.JSONDecodeError):
                    benefit = None

            similarity = cosine_similarity(cv_embedding, job['embedding'])
            matched_jobs.append({
                'id': str(job['_id']),
                'company': job['company'],
                'position_title': job['position_title'],
                'similarity_score': float(similarity),
                'benefit': "Negotiable" if benefit == 'N/A' else benefit
            })
        # print(matched_jobs)

        # Sắp xếp theo độ tương đồng giảm dần
        matched_jobs.sort(key=lambda x: x['similarity_score'], reverse=True)

        os.remove(resume_path)
        
        # Trả về top N kết quả
        return jsonify({
            "isSuccess": True,
            "data": matched_jobs[:top_n],
            "errorCode": None,
            "message": None,
        })
    
    except Exception as e:
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
    client1 = MongoClient(MONGO_URI_ATLAS,tlsCAFile=certifi.where())
    db1 = client1[DB_NAME]
    job_embedding1 = db1[JOBS_EMBEDDING]
    data = list(job_embedding1.find({}, {'company': 1, 'position_title': 1, 'model_response': 1, 'embedding': 1}))

    client2 = MongoClient(MONGO_URI)
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