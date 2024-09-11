import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from threading import Thread
import json
from bs4 import BeautifulSoup
from threading import Thread
import sqlite3
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import requests
import time

app = Flask(__name__)
CORS(app)
conversation_chain = None

db_texts = None
db_vectors = None

def initialize_data():
    """初始化从数据库读取的数据。"""
    global db_texts, db_vectors, db_links  # 添加 db_links 作为全局变量
    db_texts, db_vectors, db_links = read_vectors_from_db()
    print("数据库数据已加载。")



def get_db_connection(db_path="text_vectors.db"):
    """ Connect to the SQLite database. """
    conn = sqlite3.connect(db_path)
    return conn

def read_vectors_from_db():
    """从数据库读取文本块、对应的向量和视频链接（如果有）。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text, vector, video_time_link FROM text_vectors")
    db_data = cursor.fetchall()
    conn.close()
    texts, vectors, links = zip(*db_data)
    vectors = [np.fromstring(vector[1:-1], sep=', ') for vector in vectors]  # Convert string back to numpy array
    return texts, np.array(vectors), links

#Embedding pdf
def get_pdf_text(pdf_path):
    """从指定路径的PDF文件中提取文本。"""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # 添加或""以避免None
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def save_text_to_db(text, db_path="piazza_posts.db"):
    """将文本保存到数据库中。"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)''')
    try:
        c.execute("INSERT INTO posts (content) VALUES (?)", (text,))
        conn.commit()
        print("Text saved successfully to database.")
    except sqlite3.IntegrityError as e:
        print(f"Error saving text to database: {e}")
    finally:
        conn.close()

def run_command_line_interaction():
    global conversation_chain
    while True:
        embed_pdf = input("Do you want to embed a PDF? (y/n): ")
        if embed_pdf.lower() == 'y':
            pdf_path = input("Enter the path to the PDF: ")
            if os.path.exists(pdf_path):
                pdf_text = get_pdf_text(pdf_path)
                # 将PDF文本保存到数据库
                save_text_to_db(pdf_text)
                print("PDF text embedding process completed.")
            else:
                print("PDF file does not exist. Please check the path.")
        elif embed_pdf.lower() == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

#Get Embedding:
def get_embedding(text, model="text-embedding-3-small"):
    """
    获取给定文本的embedding向量。
    
    :param text: 要获取embedding的文本。
    :param model: 使用的embedding模型名称。
    :return: 一个包含embedding向量的列表。
    """
    openai_api_key = "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"  # 使用你的API密钥
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}
    data = {"input": text, "model": model}
    response = requests.post("https://api.openai.com/v1/embeddings", json=data, headers=headers)
    
    if response.status_code == 200:
        embedding_vector = response.json()["data"][0]["embedding"]
        return embedding_vector
    else:
        print("Failed to get embedding:", response.text)
        return None

#Compare similarity:
def compare_user_question_to_db_vectors(user_question, db_vectors, db_texts, db_links, top_k=15):
    user_question_vector = get_embedding(user_question)

    if user_question_vector is not None:
        similarities = cosine_similarity([user_question_vector], db_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 准备收集文本和链接
        top_texts = []
        top_links = []

        for index in top_indices:
            top_texts.append(db_texts[index])
            top_links.append(db_links[index])
            # 增加前后文本的逻辑
            if index > 0:
                top_texts.append(db_texts[index - 1])
                top_links.append(db_links[index - 1])
            if index < len(db_texts) - 1:
                top_texts.append(db_texts[index + 1])
                top_links.append(db_links[index + 1])

        # 检查前5个结果中的视频链接，并返回相关文本及其前后文本
        for i in range(min(5, len(top_links))):
            if top_links[i]:  # 假设检测视频链接的方式是检查链接中是否含有特定字符或域
                # 找到视频链接，返回该链接的文本及其前后文本
                linked_texts = [top_texts[i]]
                linked_texts_indices = [i]
                
                # 添加前文本
                if i > 0:
                    linked_texts.insert(0, top_texts[i - 1])
                    linked_texts_indices.insert(0, i - 1)
                
                # 添加后文本
                if i < len(top_texts) - 1:
                    linked_texts.append(top_texts[i + 1])
                    linked_texts_indices.append(i + 1)

                print(linked_texts)
                return linked_texts, top_links[i]

        # 如果在顶部结果中未找到视频链接，返回扩展的文本列表和None
        return top_texts, None
    else:
        print("Failed to vectorize user question.")
        return [], None




# 在 get_vectorstore 函数中合并文本
def get_vectorstore(texts):
    openai_api_key = "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"
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)  # 注意这里是单个字符串列表
    return vectorstore

def get_conversation_chain(vectorstore):
    openai_api_key = "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"
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def start_cli_thread():
    cli_thread = Thread(target=run_command_line_interaction)
    cli_thread.daemon = True  # 设置为守护线程,这样主程序结束时会自动结束该线程
    cli_thread.start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        start_time_total = time.time()
        user_message = request.json["message"]
        
        start_time_top_texts = time.time()
        top_texts, video_link = compare_user_question_to_db_vectors(user_message, db_vectors, db_texts, db_links)
        end_time_top_texts = time.time()
        print(f"获取top_texts耗时: {end_time_top_texts - start_time_top_texts} 秒")
        
        if video_link:
            vectorstore = get_vectorstore(top_texts)
            conversation_chain = get_conversation_chain(vectorstore)
            response = conversation_chain({"question": user_message})
            chatbot_response = response["answer"] + f"\n\nHere's a related video you might find helpful: {video_link}\n\n"
        else:
            vectorstore = get_vectorstore(top_texts)
            conversation_chain = get_conversation_chain(vectorstore)
            response = conversation_chain({"question": user_message})
            chatbot_response = response["answer"]

        end_time_total = time.time()
        print(f"从收到用户问题到给出响应总耗时: {end_time_total - start_time_total} 秒")

        print(f"Chatbot response: {chatbot_response}")
        
        return jsonify({"response": chatbot_response})
    except KeyError:
        return jsonify({"error": "Invalid request data"}), 400


if __name__ == "__main__":
    load_dotenv()
    start_cli_thread()
    initialize_data()  # 在服务器启动时加载数据
    app.run(debug=False, port=5008)