import torch
import optuna
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint
from torch.optim import AdamW
from transformers import Seq2SeqTrainer
from datasets import load_metric
from transformers import get_cosine_schedule_with_warmup

def is_cuda_available():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

def compute_wer(pred):
    metric = load_metric("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": metric.compute(predictions=pred_str, references=label_str)}

def model_init():
    model = WhisperForConditionalGeneration.from_pretrained("path_to_whisperx_model")
    peft_config = LoraConfig(task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.to(device)
    return model

def data_preprocessing():
    dataset = load_dataset("path_to_dataset")
    dataset = dataset.shuffle(seed=42)
    processor = WhisperProcessor.from_pretrained("path_to_whisperx_model")
    
    def preprocess_function(batch):
        inputs = processor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding=True)
        labels = processor(batch["text"], return_tensors="pt", padding=True).input_ids
        return {"input_features": inputs.input_features.squeeze(), "labels": labels.squeeze()}
    
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    return dataset

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
    
    training_args = TrainingArguments(
        output_dir="./whisperx_finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        fp16=True,
        logging_dir="./logs",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        optim="adamw_torch"
    )
    
    data_collator = DataCollatorForSeq2Seq(processor, padding=True)
    trainer = Seq2SeqTrainer(
        model=model_init(),
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_wer
    )
    
    trainer.train()
    result = trainer.evaluate()
    return result["eval_wer"]

if __name__ == "__main__":
    device = is_cuda_available()
    processor = WhisperProcessor.from_pretrained("path_to_whisperx_model")
    data = data_preprocessing()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print("Best hyperparameters:", study.best_params)
