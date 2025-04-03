from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import wandb
from RL_data_prep import get_critical_questions_dataset, correctness_reward_func, structure_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func
import os
from transformers.trainer_utils import get_last_checkpoint
max_seq_length = 8084 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

print("Load up Qwen 2.5 3B Instruct, and set parameters")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

print("Load up the dataset and start training")
dataset = get_critical_questions_dataset()

print("Now set up GRPO Trainer and all configurations!")
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 3000,
    max_completion_length = 2000,
    num_train_epochs = 3, # Set to 1 for a full training run
    # max_steps = 250,
    save_steps = 25,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs_trial_3",
)

print("Start training!")

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        correctness_reward_func,
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        structure_reward_func,

    ],
    args = training_args,
    train_dataset = dataset,
)
# trainer.train()
# Check if there's a checkpoint in the output directory
checkpoint = None
if os.path.exists(training_args.output_dir):
    checkpoint = get_last_checkpoint(training_args.output_dir)
    if checkpoint:
        print(f"Resuming training from checkpoint: {checkpoint}")

# Start or resume training accordingly
trainer.train(resume_from_checkpoint=checkpoint)

model.save_lora("grpo_saved_lora_trial_3")

wandb.finish()