from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

vector_db_path = "vector_db_path"
embedding_model_file = "models/all-MiniLM-L6-v2-f16.gguf"

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
query = "CR7 trả lương cao nhất thế giới vào năm nào"
response = llmchain({"query": query})
print(response['result'])
