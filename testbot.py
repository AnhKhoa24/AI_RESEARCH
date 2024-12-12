import requests
import sys
import time
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings

# Fixed paths and model
vector_db_path = "vector_db_path"

def read_vectors_db():
    """Load FAISS vector store from local storage."""
    embedding_model = GPT4AllEmbeddings()
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

class CustomAPIWrapper:
    def __init__(self, api_url, model, temperature=0.7, format="json"):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.format = format

    def query(self, question, context):
        """
        Gửi yêu cầu đến API với câu hỏi và ngữ cảnh.
        
        :param question: Câu hỏi cần trả lời.
        :param context: Ngữ cảnh liên quan.
        :return: Câu trả lời từ API.
        """
        payload = {
            "model": self.model,
            "prompt": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}",
            "suffix": "",
            "format": self.format,
            "options": {
                "temperature": self.temperature
            }
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "Không có câu trả lời.")
        except requests.RequestException as e:
            return f"Lỗi API: {str(e)}"

def create_qa_chain(api_url, model, db):
    """
    Tạo RetrievalQA chain sử dụng API chatbot.
    
    :param api_url: URL của API chatbot.
    :param model: Tên mô hình được sử dụng.
    :param db: Cơ sở dữ liệu retriever.
    :return: RetrievalQA chain.
    """
    # Tạo API wrapper
    api_wrapper = CustomAPIWrapper(api_url, model)

    # Tạo prompt mẫu
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Ngữ cảnh:\n{context}\n\n"
            "Câu hỏi: {question}"
        )
    )

    # Tạo RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=api_wrapper,  # Sử dụng CustomAPIWrapper để truy vấn API.
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),  # Tìm 3 đoạn phù hợp nhất.
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain

api_url = "http://localhost:11434/api/generate"
model = "llama3.1"
db = read_vectors_db() 
qa_chain = create_qa_chain(api_url, model, db)
# Truy vấn ví dụ
question = "biết cr7 không"
result = qa_chain.run(question)
print(result)
