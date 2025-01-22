from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # 사용할 GPU를 미지정시 에러 발생 Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!


finetune_dataset = "/home/swap/jsjeong/dataset/guanaco-llama2-1k" # 영어, 스페인어, 프랑스어, 한자 등으로 구성된 1,000행 크기 Q&A 데이터셋 parquet 파일
korean_dataset = "/home/swap/jsjeong/dataset/KoAlpaca-v1.1a/KoAlpaca_train.csv" # 한국어로 구성된 1,000행 크기 Q&A 데이터셋 csv 파일

# dataset = load_dataset(finetune_dataset, split="train")
dataset = load_dataset("csv", data_files=korean_dataset, split="train") 

# base_model = "/home/swap/jsjeong/llama_2-7b-hf"
# new_model = "llama-test" 


base_model = "/home/swap/jsjeong/llama_2-7b-chat-hf"
new_model = "llama-chat-test"

quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16) 

lora_config = LoraConfig(
    r=16, # lora attention dimension (rank, 값이 클 수록 사용하는 parameter의 수량이 증가한다)
    lora_alpha=32, # scaling : alpha/rank로 실행 -> alpha 값이 작을수록 사용하는 parameter의 수량이 증가한다
    lora_dropout=0.05,
    bias="none", # none, all, lora_only (all과 lora_only 설정 시 training 중에도 bias가 업데이트되며 같은 base-model을 사용해도 다른 결과가 나올 수 있다)
    task_type="CAUSAL_LM") # CAUSAL_LM 이외의 다른 task_type 설정 시 error 발생 


tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token # pad_token 미지정시 에러 발생

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", quantization_config=quant_config)

training_params = TrainingArguments(output_dir="/home/swap/jsjeong/results",
                                    num_train_epochs=2, 
                                    per_device_train_batch_size=64, # 64로 batch_size 설정 시 Out of memory 에러 발생
                                    optim="adamw_torch", 
                                    adam_beta1=0.9,
                                    adam_beta2=0.95,
                                    adam_epsilon=1e-5,
                                    learning_rate=2e-5, 
                                    weight_decay=0.1, 
                                    max_grad_norm=1.0, 
                                    warmup_steps=2000, 
                                    lr_scheduler_type="cosine")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text", # fine-tuning dataset의 column name
    max_seq_length=1024, # seq_length를 512로 변경할 시 batch_size 64로 fine-tuning 가능
    peft_config=lora_config,
    args=training_params,)

trainer.train() # dataset을 바꿔도 training 시간은 큰 차이 없음


trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
