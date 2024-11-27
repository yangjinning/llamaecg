import json
import numpy as np
from datasets import IterableDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig,
    EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl
)
import torch.nn.functional as F
import torch
from peft import LoraConfig, get_peft_model, TaskType
import math
import wfdb
import os
from scipy.signal import resample
import transformers
import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from datetime import datetime
import yaml
from prompter import Prompter

config = {
    'ds_mimic_train_path': "/data/PyProjects/ECG-MY/dataset/ecgqa/mimic-iv-ecg/combine/template_train.json",
    'ds_mimic_val_path': "/data/PyProjects/ECG-MY/dataset/ecgqa/mimic-iv-ecg/combine/template_valid.json",
    'ds_mimic_test_path': "/data/PyProjects/ECG-MY/dataset/ecgqa/mimic-iv-ecg/combine/template_test.json",
    'ds_ptb_train_path': "/data/PyProjects/ECG-MY/dataset/ecgqa/ptbxl/combine/template_train.json",
    'ds_ptb_val_path': "/data/PyProjects/ECG-MY/dataset/ecgqa/ptbxl/combine/template_valid.json",
    'ds_ptb_test_path': "/data/PyProjects/ECG-MY/dataset/ecgqa/ptbxl/combine/template_test.json",
    'root_mimic': "/data/PyProjects/ECG-data/MIMIC-ECG/files",
    'root_ptb': "/data/PyProjects/ECG-data/PTB-XL/physionet.org/files/ptb-xl/1.0.3",
    'model_path': "/data/PyProjects/LLM/Llama-3.1-8B",
    'path_template': "/data/PyProjects/ECG-MY/prompt_templates/ts_as_prompt.json",
    'MAX_LENGTH': 600,
    'num_train': 386396,
    'num_val': 3000,
    'seed': 42,
    'num_train_epochs': 3,
    'num_per_device_train_batch_size': 6,
    'num_gradient_accumulation_steps': 4,
    'lr': 3e-4,
    'min_lr': 1e-5,
    'warm_num_steps': 100,
    'weight_decay': 0.01,
    'num_save_per_epoch': 5,
    'ddp': True,
    "path_result": "./results/",
}

seed = config['seed']
random.seed(seed)


def load_json(ds_path, SHUFFLE=True):
    with open(ds_path, "r") as f:
        records = json.load(f)
    if SHUFFLE:
        random.shuffle(records)
    return records


def load_ecg_data(root, filepath):
    data = wfdb.rdsamp(os.path.join(root, filepath))
    signal, meta = data
    return np.array(signal)


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_tokenizer_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                     bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_quant_type='nf4'
                                                 ),
                                                 local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    DEFAULT_PAD_TOKEN = "[PAD]"
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
    return tokenizer, model


def preprocess_function(example, tokenizer, type, prompt_temp, config, chat_prompt=False, down_sample=True):
    path_template = config['path_template']
    MAX_LENGTH = config['MAX_LENGTH']
    root_ecg = config['root_mimic']

    ecg_data = load_ecg_data(root_ecg, example["ecg_path"])

    if down_sample:
        if "ts_as_prompt" in path_template:
            ecg_data = resample(ecg_data[:, 0], 250)  # only downsample on the 1st lead : 5000->250
        else:
            ecg_data = resample(ecg_data, 1000)

    question = example["question"]
    answer_text = ".".join(example["answer"])

    if "ts_as_prompt" in path_template:
        if chat_prompt:
            instruction_text = prompt_temp.generate_chat_prompt(ecg_data, question)
            answer_text += "<|eot_id|>" + tokenizer.eos_token  # special token defined in llama3.1 and 3.2
        else:
            instruction_text = prompt_temp.generate_prompt(ecg_data, question)
            answer_text += tokenizer.eos_token
    instruction = tokenizer(instruction_text)
    response = tokenizer(answer_text)

    if type in ["train", "val"]:
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    elif type == "test":
        input_ids = instruction["input_ids"]
        attention_mask = instruction["attention_mask"]
        labels = response["input_ids"]
    else:
        raise ValueError("Unrecognized type")

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    pad_input_ids = F.pad(torch.tensor(input_ids), (MAX_LENGTH - len(input_ids), 0), value=tokenizer.pad_token_id)
    pad_attention_mask = F.pad(torch.tensor(attention_mask), (MAX_LENGTH - len(attention_mask), 0), value=0)
    pad_labels = F.pad(torch.tensor(labels), (MAX_LENGTH - len(labels), 0), value=-100)

    assert type in ["train", "val", "test"]
    if type in ["train", "val"]:
        return {
            "input_ids": pad_input_ids,
            "attention_mask": pad_attention_mask,
            "labels": pad_labels
        }
    else:
        example["instruction"] = instruction_text
        example["input_ids"] = pad_input_ids
        return example


def data_generator(tokenizer, records, type, prompt_temp, config):
    for example in records:
        processed = preprocess_function(example, tokenizer, type, prompt_temp, config, chat_prompt=False,
                                        down_sample=True)
        yield processed


def main(config):
    ds_mimic_train_path = config['ds_mimic_train_path']
    ds_mimic_val_path = config['ds_mimic_val_path']
    model_path = config['model_path']
    path_template = config['path_template']
    num_train = config['num_train']
    num_val = config['num_val']
    num_train_epochs = config['num_train_epochs']
    num_per_device_train_batch_size = config['num_per_device_train_batch_size']
    num_gradient_accumulation_steps = config['num_gradient_accumulation_steps']
    lr = config['lr']
    num_save_per_epoch = config['num_save_per_epoch']
    ddp = config['ddp']
    path_result = config["path_result"]
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not os.path.exists(f'{path_result}/{current_time}/'):
        os.makedirs(f'{path_result}/{current_time}/', exist_ok=True)
    with open(f'{path_result}/{current_time}/config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

    tokenizer, model = load_tokenizer_model(model_path) #initialize model and tokenizer
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "embed_tokens",
                        "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    json_train = load_json(ds_mimic_train_path)[:num_train]
    json_val = load_json(ds_mimic_val_path)[:num_val]

    prompt_temp = Prompter(path_template)
    # Create IterableDataset
    ds_train = IterableDataset.from_generator(
        generator=data_generator,
        gen_kwargs={"tokenizer": tokenizer, "records": json_train, "type": "train", "prompt_temp": prompt_temp,
                    "config": config}
    )
    ds_val = IterableDataset.from_generator(
        generator=data_generator,
        gen_kwargs={"tokenizer": tokenizer, "records": json_val, "type": "val", "prompt_temp": prompt_temp,
                    "config": config}
    )

    num_train_samples = len(json_train)
    num_gpu = torch.cuda.device_count()
    max_steps = math.ceil(
        num_train_samples / (
                num_per_device_train_batch_size * num_gradient_accumulation_steps * num_gpu)) * num_train_epochs
    print("max steps:", max_steps)
    num_save_steps = (max_steps // num_train_epochs) // num_save_per_epoch

    training_args = TrainingArguments(
        output_dir=f"{path_result}/{current_time}/",
        per_device_train_batch_size=num_per_device_train_batch_size,
        per_device_eval_batch_size=num_per_device_train_batch_size,
        gradient_accumulation_steps=num_gradient_accumulation_steps,
        learning_rate=lr,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        optim="paged_adamw_32bit",
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=num_save_steps,
        eval_steps=num_save_steps,
        save_steps=num_save_steps,
        save_total_limit=3,
        max_steps=max_steps,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,
        remove_unused_columns=False,
        ignore_data_skip=True,
        ddp_find_unused_parameters=False if ddp else None,
        eval_accumulation_steps=100,
    )

    if ddp:
        training_args.gpus = num_gpu
        training_args.distributed_backend = "ddp"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.005)]
    )

    trainer.train()


if __name__ == "__main__":
    main(config)
