import streamlit as st

st.title("Math Chatbot")

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
import torch
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    config=quant_config, 
    device_map={"": 0}
)

llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# from peft import PeftConfig, PeftModelForCausalLM
# peft_model_id = "llama-2-7b-mlablonne-enhanced"
# config = PeftConfig.from_pretrained(peft_model_id)
# model = PeftModelForCausalLM.from_pretrained(base_model, peft_model_id)
model = base_model

message = st.text_input("You: ")
if message:
    query = message
    text_gen = pipeline(task="text-generation", model=model, tokenizer=llama_tokenizer, max_length=200)
    output = text_gen(f"<s>[INST] {query} [/INST]")
    st.text_area("Chatbot:", value=f"{output[0]['generated_text']}", height=100, disabled=True)