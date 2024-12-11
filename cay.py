from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

vector_db_path = "vectorstores/db_faiss"
pdf_data_path = "data"
embedding_model_file = "models/all-MiniLM-L6-v2-f16.gguf"

def create_db_from_text():
    """Create FAISS vector store from raw text."""
    raw_text = """
    Cristiano Ronaldo có biệt danh là CR7.
    Messi có biệt danh là râu tạ.
    Cristiano Ronaldo dos Santos Aveiro GOIH ComM (phát âm tiếng Bồ Đào Nha: sinh ngày 5 tháng 2 năm 1985) là một cầu thủ bóng đá chuyên nghiệp người Bồ Đào Nha hiện đang thi đấu ở vị trí tiền đạo và là đội trưởng của cả câu lạc bộ Saudi Pro League Al Nassr và đội tuyển bóng đá quốc gia Bồ Đào Nha.
    Cristiano Ronaldo Được đánh giá là một trong những cầu thủ vĩ đại nhất mọi thời đại, Ronaldo đã giành được vô số giải thưởng cá nhân trong suốt sự nghiệp của mình bao gồm năm Quả bóng vàng, kỷ lục ba lần nhận giải thưởng Cầu thủ nam xuất sắc nhất năm của UEFA, và bốn Chiếc giày vàng châu Âu – nhiều nhất trong số các cầu thủ châu Âu.
    Cristiano Ronaldo đã giành được 33 danh hiệu trong sự nghiệp của mình, bao gồm 7 chức vô địch quốc gia, 5 UEFA Champions League, 1 UEFA Euro và 1 UEFA Nations League.
    Cristiano Ronaldo nắm giữ các kỷ lục về số lần ra sân nhiều nhất (183), nhiều bàn thắng nhất (140) và nhiều pha kiến tạo nhất (42) ở Champions League, nhiều bàn thắng nhất ở giải vô địch châu Âu (14), nhiều bàn thắng quốc tế nhất (135) và có số lần ra sân quốc tế nhiều nhất (217).
    Cristiano Ronaldo là một trong số ít những cầu thủ đã có hơn 1.200 lần ra sân trong sự nghiệp chuyên nghiệp, nhiều nhất đối với một cầu thủ không phải thủ môn, và đã ghi hơn 900 bàn thắng chính thức trong sự nghiệp cho câu lạc bộ và đội tuyển quốc gia, giúp anh trở thành cầu thủ ghi nhiều bàn thắng nhất mọi thời đại.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(raw_text)

    embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

def load_llm(model_name):
    """Load LLM model using Ollama."""
    llm = Ollama(model=model_name)
    return llm

def create_prompt(template):
    """Create prompt for LLM."""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    """Create a RetrievalQA chain."""
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def read_vectors_db():
    """Load FAISS vector store from local storage."""
    embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Create or load the vector database
create_db_from_text()
db = read_vectors_db()

# Load the LLM
llm = load_llm("llama3.2")

# Define the prompt template
template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói là \"Thông tin này tôi chưa được cung cấp\", đừng cố tạo ra câu trả lời
{context}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

prompt = create_prompt(template)

# Create the QA chain
llmchain = create_qa_chain(prompt, llm, db)

# Query the model
query = "CR7 sinh vào năm nào"
response = llmchain({"query": query})
print(response['result'])
