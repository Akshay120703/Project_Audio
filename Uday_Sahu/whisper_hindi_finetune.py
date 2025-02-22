import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, TrainingArguments, Trainer, DataCollatorForSeq2Seq
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

# Load dataset and shuffle
dataset = load_dataset("path/to/your/dataset")  # Replace with actual dataset path
dataset = dataset.shuffle(seed=42)

# Load pre-trained model and processor
model_path = "path/to/offline/downloaded/vasishth"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# Apply LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  
    r=16,  # LoRA rank
    lora_alpha=32, 
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
device = check_device()
model.to(device)

def preprocess_data(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    return inputs

dataset = dataset.map(preprocess_data)

# Define data collator
data_collator = DataCollatorForSeq2Seq(processor)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./vasishth-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-4,
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
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_wer,
    data_collator=data_collator
)

trainer.train()
