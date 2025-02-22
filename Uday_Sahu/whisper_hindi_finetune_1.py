import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import jiwer
import optuna

def check_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

def compute_wer(pred):
    wer = jiwer.wer(pred["references"], pred["predictions"])
    return {"wer": wer}

# Load dataset, shuffle, and resample to 16kHz
def preprocess_data(batch):
    audio = batch["audio"]
    waveform, orig_sr = torch.tensor(audio["array"]), audio["sampling_rate"]
    waveform = torchaudio.transforms.Resample(orig_sr, 16000)(waveform)
    batch["audio"]["array"] = waveform.numpy()
    batch["audio"]["sampling_rate"] = 16000
    return batch

dataset = load_dataset("path/to/your/dataset")  # Replace with actual dataset path
dataset = dataset.shuffle(seed=42).map(preprocess_data)

# Load pre-trained WhisperX model and processor
model_path = "path/to/offline/downloaded/whisperx"
model = WhisperForConditionalGeneration.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)

def objective(trial):
    r = trial.suggest_int("lora_r", 8, 64)
    alpha = trial.suggest_int("lora_alpha", 16, 128)
    dropout = trial.suggest_float("lora_dropout", 0.05, 0.3)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    
    # Apply LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  
        r=r,  
        lora_alpha=alpha, 
        lora_dropout=dropout
    )
    model_tuned = get_peft_model(model, peft_config)
    device = check_device()
    model_tuned.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./whisperx-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=3,
        logging_dir="./logs",
        report_to="none",
        fp16=True,  # Mixed Precision Training (FP16)
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        save_total_limit=3,  # Keep last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="wer"
    )
    
    trainer = Trainer(
        model=model_tuned,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_wer,
        data_collator=DataCollatorForSeq2Seq(processor)
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["wer"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best Hyperparameters:", study.best_params)
