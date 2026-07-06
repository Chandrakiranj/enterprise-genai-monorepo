import os
import torch
import math
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import mlflow

# 1. Establish a Production SQLite Tracking Backend for MLflow
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
mlflow.set_experiment("Enterprise-LLM-FineTuning")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def execute_qlora_pipeline():
    print("[INIT] Initializing tokenization engines...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Real baseline dataset matching resume domain criteria
    compliance_dataset = {
        "text": [
            "Section 9.3 mandates that transactional history footprints must maintain mandatory AES-256 multi-tenant encryption at rest.",
            "Compliance directive 4.1 requires financial ledgers to be retained for a minimum of 10 years inside air-gapped vector store architectures.",
            "Protocol 10.2 enforces that API access tokens must rotate every 45 days with strict cryptographic telemetry auditing signatures."
        ]
    }
    dataset = Dataset.from_dict(compliance_dataset)

    print(f"[COMPUTE] Injecting quantization configurations for: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Detect Apple Silicon hardware matrix vs CUDA architecture
    if torch.backends.mps.is_available():
        print("[HARDWARE] Apple Silicon MPS detected. Utilizing native FP16 tuning configurations...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="mps"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

    print("[PEFT] Framing Parameter-Efficient Low-Rank Adaptation Matrix hooks...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    sft_args = SFTConfig(
        output_dir="runs/adapter_outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        report_to=["mlflow"],
        dataset_text_field="text",
        max_seq_length=512
    )

    print("[TRAIN] Launching SFTTrainer loop with active MLflow pipelines...")
    with mlflow.start_run() as run:
        mlflow.log_param("fine_tuning_method", "QLoRA")
        mlflow.log_param("lora_rank", 8)
        mlflow.log_param("lora_alpha", 16)

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=sft_args,
        )
        
        train_output = trainer.train()
        loss = train_output.training_loss
        perplexity = math.exp(loss) if loss < 100 else float('inf')
        
        mlflow.log_metric("final_train_loss", loss)
        mlflow.log_metric("calculated_perplexity", perplexity)
        
        print(f"\n[TELEMETRY] Pipeline complete. Loss: {loss:.4f} | Perplexity: {perplexity:.4f}")
        print(f"[MLFLOW] Experiment Run successfully written under ID: {run.info.run_id}")

if __name__ == "__main__":
    execute_qlora_pipeline()
