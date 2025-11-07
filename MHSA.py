import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, key, value, causal_mask=None, past_key_value=None):
        batch_size = query.size(0)

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        
        # Missing .transpose(1, 2)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        print('query', query.shape) # torch.Size([2, 4, 10, 5])


        # ------------------------------
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, key], dim=2)
        # ------------------------------

        # query = apply_rope(query)
        # key = apply_rope(key)

        # Wrong, not self.num_heads, is self.head_dim
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim)) # [seq_len,head_dim] @ [head_dim,seq_len], the computation per head is carried out independently.
        print('attention_scores', attention_scores.shape) # torch.Size([2, 4, 10, 10])


        # ------------------------------
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9
        print('attention_scores\n', attention_scores) # torch.Size([2, 4, 10, 10])
        # ------------------------------

        # Missing softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        print('attention_scores after softmax\n', attention_scores) # torch.Size([2, 4, 10, 10])

        output = torch.matmul(attention_probs, value)
        print('output', output.shape) # torch.Size([2, 4, 10, 5])

        # Missing combine MH
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.head_dim)
        
        output = self.o_linear(output)
        print(output.shape)

        return output




if __name__ == '__main__':
    batch_size = 1
    seq_len = 5 
    hidden_size = 10
    num_heads = 2
    query = torch.zeros(batch_size, seq_len, hidden_size)
    key = torch.zeros(batch_size, seq_len, hidden_size)
    value = torch.zeros(batch_size, seq_len, hidden_size)

    causal_mask = torch.triu(torch.ones(hidden_size//num_heads, hidden_size//num_heads), diagonal=1)
    print(causal_mask)

    mhsa = MultiHeadAttention(hidden_size, num_heads)

    output = mhsa(query, key, value, causal_mask)