# como n consegui rodar com o NeMo, vou fazer com pytorch fds
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os, json
import matplotlib.pyplot as plt
from datetime import datetime

class SimpleTextDataset(Dataset):
    def __init__(self, data_prefix, tokenizer, seq_length, split='train'):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        bin_file = f"{data_prefix}_{split}_text_document.bin"
        idx_file = f"{data_prefix}_{split}_text_document.idx"
        
        if os.path.exists(bin_file) and os.path.exists(idx_file):
            self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
  
    def __len__(self):
        return max(1, (len(self.data) - self.seq_length) // self.seq_length)
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        
        if end_idx > len(self.data):
            tokens = list(self.data[start_idx:])
            while len(tokens) < self.seq_length + 1:
                tokens.append(self.tokenizer.eos_token_id)
        else:
            tokens = self.data[start_idx:end_idx]
        
        inputs = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        
        return inputs, targets

def train_model():
    seq_length = 512
    batch_size = 2
    lr = 6e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    tokenizer = AutoTokenizer.from_pretrained("gpt2") # ele me deu um erro na hora de usar o local, não entendi pq, vou importar normal
    tokenizer.pad_token = tokenizer.eos_token
    
    # como eu tava tentando usar o NeMo, meu data está em .bin que era como o Megatron pedia
    train_dataset = SimpleTextDataset("data/train_gpt", tokenizer, seq_length, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=seq_length,
        n_ctx=seq_length,
        n_embd=256,   
        n_layer=2,       
        n_head=4,       
        n_inner=1024,   
    )
    model = GPT2LMHeadModel(config)
    model.to(device)
    
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # WSD que o diogo pediu, vou setar por aqui, pois no NeMo, já tinha a função, aqui vou ir por lógica mesmo
    warmup_steps = 15        # warmup que vai ser linear
    constant_steps = 30      # Constant  
    total_training_steps = 80  
    decay_steps = total_training_steps - warmup_steps - constant_steps  # Decay
    
    print(f"WSD: warmup({warmup_steps}) -> constant({constant_steps}) -> decay({decay_steps}) = total({total_training_steps})")
    
    def get_lr(step):
        #obvio que tem formas melhor de fazer isso, mas como só vou replicar em um base mesmo, acho que ta bom, se bem que 
        #dava para colocar em menos dois if, porém ta bom
        if step < warmup_steps:
            return step / warmup_steps
        elif step < warmup_steps + constant_steps:
            return 1.0
        elif step < total_training_steps:
            decay_step = step - warmup_steps - constant_steps
            progress = decay_step / decay_steps 
            return 0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2
        else:
            return 0.1
    
    model.train()
    step = 0
    train_losses = []
    learning_rates = []
    steps_list = []
    
    # vou rodar baseado em steps, até pq só quero ver se funciona mesmo
    ## tirei esse código de uma implementação da NET, só mudei para add o WSD
    while step < total_training_steps:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if step >= total_training_steps:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # ponto chave learning rate (WSD)
            lr_mult = get_lr(step)
            current_lr = lr * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_value = loss.item()
            train_losses.append(loss_value)
            learning_rates.append(current_lr)
            steps_list.append(step)
            
            # vou add uns print só para ver se ta funcionando
            if step < warmup_steps:
                phase = "WARMUP"
            elif step < warmup_steps + constant_steps:
                phase = "CONSTANT"
            else:
                phase = "DECAY"
            
            if step % 5 == 0:  
                print(f"{step:3d} [{phase:8s}] loss: {loss_value:.4f}, LR: {current_lr:.6f}")
            
            step += 1
    
    print(f"\nfoi! total de {step} step")
    
    # preciso de grafico para mostrar para o diogo, igual tem no:
    #https://www.researchgate.net/figure/llustration-of-learning-rate-curves-for-Cosine-WSD-and-our-Power-schedulers_fig1_383428116
    create_training_plots(steps_list, train_losses, learning_rates, warmup_steps, constant_steps, total_training_steps)
    
    training_data = {
        'steps': steps_list,
        'losses': train_losses,
        'learning_rates': learning_rates,
        'config': {
            'warmup_steps': warmup_steps,
            'constant_steps': constant_steps,
            'decay_steps': decay_steps,
            'total_training_steps': total_training_steps,
            'max_lr': lr,
            'min_lr': lr * 0.1,
            'batch_size': batch_size,
            'seq_length': seq_length
        },
        'wsd_phases': {
            'warmup': f'Steps 0-{warmup_steps}',
            'constant': f'Steps {warmup_steps}-{warmup_steps + constant_steps}',
            'decay': f'Steps {warmup_steps + constant_steps}-{total_training_steps}'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs("wsd_demo_logs", exist_ok=True)
    model.save_pretrained("wsd_demo_logs/final_model")
    tokenizer.save_pretrained("wsd_demo_logs/final_model")
    
    with open("wsd_demo_logs/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print("Modelo salvo em wsd_demo_logs/final_model")
    print("Dados de treinamento salvos em wsd_demo_logs/training_data.json")
    print("Gráficos salvos em wsd_demo_logs/")


## vou pedir para o gpt criar os gráficos 
def create_training_plots(steps, losses, learning_rates, warmup_steps, constant_steps, total_steps):
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    colors = {'warmup': 'orange', 'constant': 'green', 'decay': 'red'}
    ax1.plot(steps, losses, 'b-', linewidth=2, alpha=0.8, label='Loss por step')
    if len(losses) > 5:
        window_size = min(7, len(losses) // 3)
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = steps[window_size-1:]
        ax1.plot(smoothed_steps, smoothed_losses, 'r-', linewidth=3, label=f'Média móvel ({window_size} steps)')
    if len(steps) > 0:
        ax1.axvspan(0, warmup_steps, alpha=0.2, color=colors['warmup'], label='Warmup Phase')
        ax1.axvspan(warmup_steps, warmup_steps + constant_steps, alpha=0.2, color=colors['constant'], label='Constant Phase')
        ax1.axvspan(warmup_steps + constant_steps, total_steps, alpha=0.2, color=colors['decay'], label='Decay Phase')
    
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss ao longo do tempo (com fases WSD)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, learning_rates, 'g-', linewidth=3, label='Learning Rate', marker='o', markersize=2)
    ax2.axvline(x=warmup_steps, color=colors['warmup'], linestyle='--', linewidth=2, alpha=0.8, label=f'Fim Warmup (step {warmup_steps})')
    ax2.axvline(x=warmup_steps + constant_steps, color=colors['decay'], linestyle='--', linewidth=2, alpha=0.8, label=f'Início Decay (step {warmup_steps + constant_steps})')

    ax2.axvspan(0, warmup_steps, alpha=0.15, color=colors['warmup'])
    ax2.axvspan(warmup_steps, warmup_steps + constant_steps, alpha=0.15, color=colors['constant'])
    ax2.axvspan(warmup_steps + constant_steps, total_steps, alpha=0.15, color=colors['decay'])

    ax2.text(warmup_steps/2, max(learning_rates)*0.9, 'WARMUP', ha='center', fontsize=10, fontweight='bold', color=colors['warmup'])
    ax2.text(warmup_steps + constant_steps/2, max(learning_rates)*0.9, 'CONSTANT', ha='center', fontsize=10, fontweight='bold', color=colors['constant'])
    ax2.text(warmup_steps + constant_steps + (total_steps - warmup_steps - constant_steps)/2, max(learning_rates)*0.9, 'DECAY', ha='center', fontsize=10, fontweight='bold', color=colors['decay'])
    
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('WSD Learning Rate Schedule (Warmup → Stable → Decay)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, total_steps)

    scatter = ax3.scatter(learning_rates, losses, c=steps, cmap='viridis', alpha=0.7, s=20, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Learning Rate', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Loss vs Learning Rate (colorido por step)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Step', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)

    os.makedirs("wsd_demo_logs", exist_ok=True)
    plt.savefig("wsd_demo_logs/training_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig("wsd_demo_logs/training_plots.pdf", bbox_inches='tight')
    
    plt.show()
    
    print("Gráficos salvos como:")
    print("- wsd_demo_logs/training_plots.png")
    print("- wsd_demo_logs/training_plots.pdf")

if __name__ == "__main__":
    train_model()