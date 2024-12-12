from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

def load_file(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    # Sử dụng pipe để kết nối các bước
    sequence = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type ="stuff",
        retriever = db.as_retriever(search_kwargs ={"k":3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt':prompt} 
    )
    return sequence
def readVetorsDB():
    embedding_model = GPT4AllEmbeddings(model_file= "models\all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

db = readVetorsDB()
llm = load_file(model_file)
# print(f"Number of documents in FAISS: {db.index.ntotal}")

template = """<|im_start|>system
\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói là "Thông tin này tôi chưa được cung cấp", đừng cố tạo ra câu trả lời
{context}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

prompt = create_prompt(template)
llmchain = create_qa_chain(prompt, llm, db)
query = "thần tượng của cr7 là ai "
response = llmchain.invoke({"query": query})
print(response)