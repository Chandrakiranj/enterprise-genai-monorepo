import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def execute_qlora_tuning():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[TRAIN] Downloading base target model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Configure Low-Rank Adaptation (LoRA) hyperparameters
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    print("[TRAIN] LoRA adapter matrices successfully attached to frozen base layers.")
    model.print_trainable_parameters()

    # Load custom domain data
    with open("src/training/dataset.json", "r") as f:
        data = json.load(f)

    def tokenize_function(example):
        text = f"Instruction: {example['instruction']}\nOutput: {example['output']}"
        inputs = tokenizer(text, truncation=True, max_length=128, padding="max_length")
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_data = [tokenize_function(item) for item in data]

    # Configure training run parameters
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_steps=1,
        learning_rate=2e-4,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    print("[START] Running fine-tuning loops...")
    trainer.train()
    
    # Export fine-tuned weights
    model.save_pretrained("./src/training/secure_compliance_adapter")
    print("[SUCCESS] Fine-tuning complete. Adapter weights preserved locally.")

if __name__ == "__main__":
    execute_qlora_tuning()
