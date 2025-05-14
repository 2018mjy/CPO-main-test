# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer
import argparse
import json
# Define and parse arguments.


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--percentage', type=float, default=1)
    args.add_argument('--output_dir', type=str, default="./results_hotpot_7b_base")
    args.add_argument('--base_model', type=str, default="")
    args.add_argument('--wandb_name', type=str, default='dpo_llama_2')
    args.add_argument('--dataset', type=str, default='hotpotqa_7b_data.json')
    args.add_argument('--bs', type=int, default=4)
    args.add_argument('--lora_r', type=int, default=8)
    args.add_argument('--mixed', type=bool, default=False)
    args.add_argument('--randomseed', type=int, default=False)
    args = args.parse_args()
    return args

args = parse_args()
pct = args.percentage
bs = args.bs
r = args.lora_r
mixed = args.mixed
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script. 
    - 参数封装
    """

    # data parameters
    beta: Optional[float] = field(default=0.2, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters - 封装了训练相关的超参数
    base_model: Optional[str] = field(
        default=args.base_model,
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.00, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})
    # 是否混合数据集
    mixed: Optional[bool] = field(default=mixed, metadata={"help": "whether training with mixed datasets"})
    per_device_train_batch_size: Optional[int] = field(default=bs, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=bs, metadata={"help": "eval batch size per device"})
    randomseed: Optional[int] = field(default=0, metadata={"help": "randomseed"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    # 按比例采样
    percentage: float = field(default=1.0, metadata={"help": "Description of the percentage parameter."})
    bs: float = field(default=4, metadata={"help": "Description of the batch_size parameter."})
    
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=r, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=900, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=300, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    wandb_name: Optional[str] = field(default="dpo_llama_2", metadata={"help": "the output directory"})
    dataset: Optional[str] = field(default="hotpotqa_7b_data.json", metadata={"help": "the output directory"})
    
    output_dir: Optional[str] = field(default="./results_hotpot_7b_base", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})

    # instrumentation 
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.
    该函数用于从 Hugging Face 的 lvwerra/stack-exchange-paired 数据集加载 Stack Exchange 问答配对数据，并将其转换为 DPO（Direct Preference Optimization）训练所需的格式
    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names
    # 如果 sanity_check 为 True，则只选取前 1000 条数据用于快速检查。
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    # 解析参数
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # 1. load a pretrained model 加载预训练模型
    print('=====load a pretrained model====')
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, # 设置为 bfloat16 精度，节省显存
        # load_in_4bit=True,
    )

    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset 加载配对数据集
    print('====Load the Stack-exchange paired dataset====')
    ori_dataset = []
    # 如果不是混合数据集（ args.mixed == False ），则从指定的 json 文件加载配对数据（通常是 prompt, chosen, rejected 三元组）
    if args.mixed == False:
        with open(args.dataset, 'r') as f:
            ori_dataset.extend(json.load(f))
    # 根据 pct 参数（采样比例）对数据集进行采样，支持全量或随机采样，便于实验和大规模训练  
    len_data = round(len(ori_dataset)*pct)
    if pct == 1:
        ori_dataset = ori_dataset[:len_data]
    else:
        import random
        random.seed(args.randomseed) # 设置随机种子
        random_numbers = random.sample(range(0, len(ori_dataset)), len_data)
        selected_dataset = []
        for i, d in enumerate(ori_dataset):
            if i in random_numbers:
                selected_dataset.append(d)
            # else:
            #     d['chosen'], d['rejected'] = d['rejected'], d['chosen'] 
            #     selected_dataset.append(d)
        ori_dataset = selected_dataset
    # if 'negative' in args.output_dir:
    #     ori_dataset = ori_dataset[:3000]
    print('number of paired_data: ' + str(len(ori_dataset)))
    # 将数据转换为适合的字典格式，再转为 HuggingFace 的 Dataset 对象
    data_dict = {key: [item[key] for item in ori_dataset] for key in ori_dataset[0]}
    # 创建datasets.Dataset对象
    dataset = Dataset.from_dict(data_dict)
    # 划分为训练集和测试集（10%做测试）
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    # 计算 warmup 步数，保证不会太小
    warmup_steps = round(0.1*len(train_dataset)/(4*bs))
    if warmup_steps < 10:
        warmup_steps = 10
    # 3. Load evaluation dataset 直接取 Dataset 的 test 部分作为评估集
    print('====Load evaluation dataset====')
    eval_dataset =dataset['test']


    # 4. initialize training arguments: 初始化训练参数（包括 batch size、最大步数、保存/评估频率、学习率、优化器类型、是否使用 bf16、日志设置等）
    print('====initialize training arguments:====')
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        max_steps=round(len(train_dataset)/(4*bs))*3,
        logging_steps=script_args.logging_steps,
        # save_steps=script_args.save_steps,
        save_steps=round(len(train_dataset)/(4*bs)*0.5),
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=args.wandb_name,
    )
    # 配置 LoRA 微调参数（如 r、alpha、dropout），并指定应用 LoRA 的模块（如 q_proj、v_proj 等），以减少微调参数量
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    print('====initialize the DPO trainer====')
    dpo_trainer = DPOTrainer(
        model,
        None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    print('====train====')
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    print('====save====')
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
