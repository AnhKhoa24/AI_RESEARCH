# from llama_cpp import Llama

# def load_model_with_llama(model_path):
#     llm = Llama(
#         model_path=model_path,
#         n_gpu_layers=1,  # Số lớp sẽ được load lên GPU
#         n_ctx=4096,      # Kích thước context
#         f16_kv=True,     # Sử dụng Half-precision để tăng tốc
#         logits_all=False,
#         use_mmap=True,   # Sử dụng memory-mapped files
#         use_mlock=True   # Giữ mô hình trong RAM (giảm tải lên disk)
#     )
#     return llm

# model_path = "models/vinallama-7b-chat_q5_0.gguf"
# llm = load_model_with_llama(model_path)

# # Gọi trực tiếp
# result = llm("1+1 bằng mấy!")
# print(result)

################################################
from llama_cpp import Llama

# Load the model
model_path = "models/vinallama-7b-chat_q5_0.gguf"
llm = Llama(model_path=model_path, n_gpu_layers=1, n_ctx=4096)

# Gọi model với temperature
response = llm(
    "1+1 bằng mấy!",
    max_tokens=1024,
    temperature=0.01,   # Điều chỉnh temperature ở đây
    top_p=0.9,          # Điều chỉnh top-p sampling
    top_k=40,           # Điều chỉnh top-k sampling
    repeat_penalty=1.2  # Giảm lặp từ
)

print(response["choices"][0]["text"])
