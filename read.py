from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

vector_db_path = "vectorstores/db_faiss"
pdf_data_path = "data"

def create_db_from_text():
    raw_text = """
    Cristiano Ronaldo có biệt danh là CR7.
    Messi có biệt danh là râu tạ.
    Cristiano Ronaldo dos Santos Aveiro GOIH ComM (phát âm tiếng Bồ Đào Nha: [kɾiʃˈtjɐnu ʁɔˈnaldu]; sinh ngày 5 tháng 2 năm 1985) là một cầu thủ bóng đá chuyên nghiệp người Bồ Đào Nha hiện đang thi đấu ở vị trí tiền đạo và là đội trưởng của cả câu lạc bộ Saudi Pro League Al Nassr và đội tuyển bóng đá quốc gia Bồ Đào Nha.
    Cristiano Ronaldo Được đánh giá là một trong những cầu thủ vĩ đại nhất mọi thời đại, Ronaldo đã giành được vô số giải thưởng cá nhân trong suốt sự nghiệp của mình bao gồm năm Quả bóng vàng,[ghi chú 3] kỷ lục ba lần nhận giải thưởng Cầu thủ nam xuất sắc nhất năm của UEFA, và bốn Chiếc giày vàng châu Âu – nhiều nhất trong số các cầu thủ châu Âu.
    Cristiano Ronaldo đã giành được 33 danh hiệu trong sự nghiệp của mình, bao gồm 7 chức vô địch quốc gia, 5 UEFA Champions League, 1 UEFA Euro và 1 UEFA Nations League.
    Cristiano Ronaldo nắm giữ các kỷ lục về số lần ra sân nhiều nhất (183), nhiều bàn thắng nhất (140) và nhiều pha kiến tạo nhất (42) ở Champions League, nhiều bàn thắng nhất ở giải vô địch châu Âu (14), nhiều bàn thắng quốc tế nhất (135) và có số lần ra sân quốc tế nhiều nhất (217). 
    Cristiano Ronaldo là một trong số ít những cầu thủ đã có hơn 1.200 lần ra sân trong sự nghiệp chuyên nghiệp, nhiều nhất đối với một cầu thủ không phải thủ môn, và đã ghi hơn 900 bàn thắng chính thức trong sự nghiệp cho câu lạc bộ và đội tuyển quốc gia, giúp anh trở thành cầu thủ ghi nhiều bàn thắng nhất mọi thời đại."""
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size = 500,
        chunk_overlap= 50,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    embedding_model = GPT4AllEmbeddings(model_file= "models\all-MiniLM-L6-v2-f16.gguf")

    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)

    # In ra số phần và nội dung từng phần
    print(f"Số phần: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Phần {i + 1}:\n{chunk}\n")

    return db

def create_db_from_files():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter( separator=".",chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    embedding_model = GPT4AllEmbeddings(model_file= "models\all-MiniLM-L6-v2-f16.gguf")

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)

  # In ra số phần và nội dung từng phần
    print(f"Số phần: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Phần {i + 1}:\n{chunk}\n")

    return db

def create_db_from_docx():
    loader = DirectoryLoader(pdf_data_path, glob="*.docx", loader_cls=Docx2txtLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator=".", chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embedding_model = GPT4AllEmbeddings()
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local("vector_db_path") 
    print(f"Số phần: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Phần {i + 1}:")
        print(chunk)
        print()

    return db

create_db_from_docx()