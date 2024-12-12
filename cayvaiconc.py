import time
import sys
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Fixed paths and model
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
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def read_vectors_db():
    """Load FAISS vector store from local storage."""
    embedding_model = GPT4AllEmbeddings()
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def print_slow(text, delay=0.1):
    """Prints the text slowly, simulating typing."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def create_prompt_with_history(template, history):
    """Create prompt that includes the conversation history."""
    context = "\n".join(history)  # Join all previous exchanges
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt.format(context=context, question="{question}")

# Function to simulate a chat conversation
def chat():
    # Load the FAISS database and LLM
    db = read_vectors_db()
    llm = load_llm("llama3.1")

    # Define the prompt template
    template = """<|im_start|>system
    Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói là \"Thông tin này tôi chưa được cung cấp\", đừng cố tạo ra câu trả lời, nếu câu hỏi là những lời chào hoặc biểu cảm hãy trả lời lại theo ý bạn
    {context}
    <|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """

    prompt = create_prompt(template)
    # Create the QA chain
    llmchain = create_qa_chain(prompt, llm, db)
    # Maintain the conversation history
    history = []

    print("Chào bạn! Tôi là trợ lý AI. Hãy hỏi tôi bất kỳ câu hỏi nào.")
    
    while True:
        # Get user input
        user_input = input("Bạn: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Tạm biệt!")
            break
        history.append(f"User: {user_input}")
        prompt_with_history = create_prompt_with_history(template, history)
        response = llmchain({"query": user_input})
        
        model_response = response['result']
        history.append(f"Assistant: {model_response}")
        print_slow(model_response)

# Start the chat
chat()
