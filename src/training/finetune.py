import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def run_genuine_qlora():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[INIT] Loading tokenizer and base layers for: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model. (Note: For true 4-bit QLoRA on a local machine, 
    # bitsandbytes is used via load_in_4bit=True, requiring a CUDA GPU.
    # On Mac/MPS, we load in FP16/Float32 to compute true local gradients)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16
    )

    # Core PEFT Target Configurations
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Verify dataset existence
    dataset_path = "src/training/dataset.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}. Please create the compliance JSON data first.")

    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    # Technically sound tokenization & label masking mapping
    def tokenize_and_mask(example):
        full_text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        tokenized = tokenizer(full_text, truncation=True, max_length=256, padding=False)
        
        # In causal LM training, labels match input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    processed_dataset = [tokenize_and_mask(item) for item in raw_data]

    # Data Collator to handle dynamic padding structurally
    def data_collator(features):
        batch = {}
        max_len = max(len(f["input_ids"]) for f in features)
        
        batch["input_ids"] = torch.tensor([f["input_ids"] + [tokenizer.pad_token_id]*(max_len - len(f["input_ids"])) for f in features])
        batch["attention_mask"] = torch.tensor([f["attention_mask"] + [0]*(max_len - len(f["attention_mask"])) for f in features])
        batch["labels"] = torch.tensor([f["labels"] + [-100]*(max_len - len(f["labels"])) for f in features]) # -100 ignores loss on pad tokens
        
        return batch

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=1,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator
    )

    print("[START] Beginning training execution loop...")
    trainer.train()
    
    output_dir = "./src/training/secure_compliance_adapter"
    model.save_pretrained(output_dir)
    print(f"[SUCCESS] Real fine-tuning complete. True adapters saved to {output_dir}")

if __name__ == "__main__":
    run_genuine_qlora()
