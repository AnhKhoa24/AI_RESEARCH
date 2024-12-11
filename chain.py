from langchain_community.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate

model_file = "models/vinallama-7b-chat_q5_0.gguf"

def load_file(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

def create_simple_chain(prompt, llm):
    # Sử dụng pipe để kết nối các bước
    sequence = prompt | llm
    return sequence

template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
prompt_template = create_prompt(template)
llm_load = load_file(model_file)
chain = create_simple_chain(prompt_template, llm_load)

# Gọi chuỗi hành động với đầu vào
question = "1+1 bằng mấy!"
result = chain.invoke({"question": question})
print(result)
