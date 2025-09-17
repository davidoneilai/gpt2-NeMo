import torch
from transformers import AutoTokenizer
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule

if __name__ == "__main__":
    seq_length = 512
    global_batch_size = 4
    hf_tok = AutoTokenizer.from_pretrained("gpt2_tokenizer/tokenizer.json", use_fast=True)
    
    data = PreTrainingDataModule(
        tokenizer=hf_tok,                
        micro_batch_size=global_batch_size,
        seq_length=seq_length,
        num_workers=2,
        # prefixos dos arquivos Megatron
        train_data_prefix=["data/train_gpt_text_document"],
        validation_data_prefix=["data/eval_gpt_text_document"],
        # flags padrão do Megatron:
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        rampup_batch_size=None,
        drop_last=True,
        persistent_workers=True,
    )
    
    # === GPT pequenininho ===
    gpt_cfg = llm.GPTConfig(
        num_layers=2,
        hidden_size=256,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        seq_length=seq_length
    )
    model = llm.GPTModel(gpt_cfg)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    # otimizador + WSD (3 fases) 
    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=6e-4,                 # lr base para a Fase 2
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=50,         # Fase 1: WARMUP -> sobe até 6e-4
            constant_steps=100,      # Fase 2: CONSTANT -> segura 6e-4
            min_lr=6e-5,             # Fase 3: DECAY -> cosine de 6e-4 para 6e-5
        ),
    )

    # === trainer: garanta max_steps >= warmup + constant + decay ===
    # aqui vou deixar 400 steps para ter sobra de 'decay'
    trainer = nl.Trainer(
        devices=1,
        accelerator="gpu",
        max_steps=400,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    #logger = nl.NeMoLogger(log_dir="wsd_demo_logs") --> NÃO FUNCIONAAAAA AAA

    # === treino ===
    llm.train(model=model, data=data, trainer=trainer, log=logger, tokenizer="data", optim=optim)