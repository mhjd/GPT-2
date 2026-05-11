from dataclasses import dataclass
import tiktoken
import math
import torch.nn as nn
import torch.nn.functional as F
import torch

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T 

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens.")
        print(f"1 epoch = {len(self.tokens) / (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # input and targets
        x = (buf[:-1]).view(B, T) 
        y = (buf[1:].view(B, T))

        # increment position of the tensor
        self.current_position += B*T 
        # if next batch out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Obtain Q K V from the token
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Combine info coming from the different head attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # allow to mask the future
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1 , config.block_size, config.block_size))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.noraml_(module.weight, mean=0.0, std=0.02)
    def forward(self, x):
        # batch size, sequence length, dimensions (n_embd)
        B, T, C = x.size()
        qkv = self.c_attn(x)
        
        q, k, v = qkv.split(self.n_embd, dim=2)
        # B, T, C -> B, number head, T, head size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2 ,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = att @ v 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # (B, T, C) -> (B, T, C)
        y = self.c_proj(y)
        return y

# Feed Forward Network
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # Expand in a larger vector space
        x = self.c_fc(x)
        # non linearity applied, to learn 
        x = self.gelu(x)
        # Projection in the original vector space
        x = self.c_proj(x)
        return x
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 # size of embeddings

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # the input embedding table is the projection matrix of the output
        # at the beginning : token -> token vector 
        # at the end : predicted vector -> comparison with vector tokens
        # in both case, we use the table of token vector
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    def  _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls):
        """Loads pretrained smallest GPT-2 model weights from huggingface"""
        from transformers import GPT2LMHeadModel

        model_type = "gpt2"
        print("loading weights from pretrained gpt: gpt2")
        config_args = dict(n_layer=12, n_head=12, n_embd=768)
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)

        # exclude the causal attention mask from state dict keys 
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a hf model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # excluding hf causal attention mask buffers 
        sd_keys_hf = sd_hf.keys()
        excluded_suffixes = (".attn.masked_bias", ".attn.bias")
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(excluded_suffixes)]

        # Weights that must be transposed when loading from Hugging Face/OpenAI GPT-2
        # because our model uses nn.Linear while the checkpoints use GPT-2's Conv1D format
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # Transpose weights stored in GPT-2's Conv1D format before copying into nn.Linear 
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # weight decay => help to regularize, not having too heavy weights
        # so, we are limiting the overfitting
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # splitting parameters between which should be weight-decayed, and those which shouldn't 
        # weight decaying only 2 dimensional parameters, so embeddings and matrices that participating in Linear
        decay_params= [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nodecay_params, 'weight_decay' : 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:n} parameters")
        print(f"num non decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:n} parameters")
        import inspect
        # fused allow updating all parameters with adam in one time instead doing a for loop
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
        
import time 

num_return_sequences = 5
max_length = 30


# total_batch_size = 524288 # 2**19, ~0.5M in number of tokens
# B = 2 # micro batch size
# T = 1024 # sequence length
# assert total_batch_size % (B * T) == 00, "make sure total_batch_size is divisible by B * T"
# grad_accum_steps = total_batch_size // (B * T)
# print(f"total desired batch size: {total_batch_size}")
# print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=16, T=1024)
# train_loader = DataLoaderLite(B=16, T=1024)

# optimisation of matrix multiplication of Linear layer
torch.set_float32_matmul_precision('high')

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)


model = GPT(GPTConfig(vocab_size=50304))  

model.to(device)
import os
if os.environ.get("ENABLE_TORCH_COMPILE", "1") == "1":
    model = torch.compile(model)

average_tps = 0
max_steps = 50
warmup_steps = 10
max_lr = 6e-4
min_lr = max_lr * 0.1
def get_lr(it):
    # warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # after the optimisation
    if it > max_steps:
        return min_lr
    # in between
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    # calculating the norm the parameters
    # so a potention high loss don't disrupt the model with a high gradient
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 )
    # determine and set dthe learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    # time difference in ms
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    average_tps += tokens_per_sec
    print(f"step {step}, loss: {loss.item()}, lr : {lr:.4f}, norm : {norm:.4f},  dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
print(f"Average number of tok/sec : {average_tps / max_steps}")

import sys; sys.exit(0)
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, all of that is because of the")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position (so the next word)
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # get the top 50 probabiiltes 
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top 50 probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # retrieve the corresponding indexes
        xcol = torch.gather(topk_indices, - 1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
