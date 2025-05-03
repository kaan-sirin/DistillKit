import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
data = torch.load("generated_tokens/sassy/teacher_random_logits_320_R300.pt", map_location="cpu")   
caesar_sequence = data[10][:10]
for idx, token_logit_pair in enumerate(caesar_sequence):
    print("--- " + "token " + str(idx) + " " +"-"*20)
    token_logit_pair.sort(key=lambda x: x[1], reverse=True)
    num_tokens = len(token_logit_pair)
    for token, prob in token_logit_pair[:3]:
        print(tokenizer.decode(token), format(prob, ".3f"))
    if num_tokens > 3:
        print(f"*{num_tokens-3} more tokens...*")
        
