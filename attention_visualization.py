import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# 创建一个简单的例子
batch_size = 1
num_heads = 1
seq_len = 4

# 创建一个示例的 attention_scores
# 形状: [batch_size, num_heads, seq_len_q, seq_len_k]
attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len) * 2

print("Attention Scores shape:", attention_scores.shape)
print("Dimension explanation: [batch_size, num_heads, seq_len_query, seq_len_key]")
print(f"Current shape: [{batch_size}, {num_heads}, {seq_len}, {seq_len}]")
print("\nOriginal Attention Scores (batch and head dims removed):")
scores_2d = attention_scores[0, 0]
print(scores_2d)

# Apply softmax on dim=-1
attention_probs = F.softmax(attention_scores, dim=-1)
probs_2d = attention_probs[0, 0]

print("\nAttention Probs after Softmax:")
print(probs_2d)
print("\nSum of each row (should all be 1.0):")
print(probs_2d.sum(dim=-1))

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Figure 1: Attention Scores Matrix Visualization
ax1 = axes[0, 0]
im1 = ax1.imshow(scores_2d.numpy(), cmap='RdYlBu_r', aspect='auto')
ax1.set_title('Attention Scores Matrix\n(Before Softmax)', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Key Position (seq_len_k) →', fontsize=12)
ax1.set_ylabel('Query Position (seq_len_q) ↓', fontsize=12)

# 添加数值
for i in range(seq_len):
    for j in range(seq_len):
        text = ax1.text(j, i, f'{scores_2d[i, j].item():.2f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im1, ax=ax1, label='Score Value')

# Add arrows to show dim=-1
for i in range(seq_len):
    ax1.annotate('', xy=(seq_len-0.5, i), xytext=(-0.7, i),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(-1.5, seq_len/2 - 0.5, 'Softmax\nalong this\ndirection\n(dim=-1)', 
         fontsize=11, color='red', fontweight='bold', 
         ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Figure 2: Attention Probs Matrix Visualization
ax2 = axes[0, 1]
im2 = ax2.imshow(probs_2d.numpy(), cmap='YlOrRd', aspect='auto')
ax2.set_title('Attention Probs Matrix\n(After Softmax, each row sums to 1)', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Key Position (seq_len_k) →', fontsize=12)
ax2.set_ylabel('Query Position (seq_len_q) ↓', fontsize=12)

# 添加数值
for i in range(seq_len):
    for j in range(seq_len):
        text = ax2.text(j, i, f'{probs_2d[i, j].item():.3f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im2, ax=ax2, label='Probability Value')

# Figure 3: Detailed explanation of one row
ax3 = axes[1, 0]
query_idx = 1  # Select the 2nd query position
scores_row = scores_2d[query_idx].numpy()
probs_row = probs_2d[query_idx].numpy()

x = np.arange(seq_len)
width = 0.35

bars1 = ax3.bar(x - width/2, scores_row, width, label='Scores before Softmax', alpha=0.8, color='steelblue')
bars2 = ax3.bar(x + width/2, probs_row, width, label='Probs after Softmax', alpha=0.8, color='coral')

ax3.set_xlabel('Key Position', fontsize=12)
ax3.set_ylabel('Value', fontsize=12)
ax3.set_title(f'Attention from Query{query_idx} to All Key Positions\n(This is Why We Apply Softmax on dim=-1)', 
              fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels([f'Key{i}' for i in range(seq_len)])
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add sum annotation
sum_text = f'Sum of probs after Softmax: {probs_row.sum():.4f} ≈ 1.0'
ax3.text(0.5, 0.95, sum_text, transform=ax3.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Figure 4: Conceptual Diagram
ax4 = axes[1, 1]
ax4.axis('off')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

# Title
ax4.text(5, 9.5, 'Why Softmax on dim=-1?', fontsize=16, fontweight='bold', ha='center')

# Draw matrix diagram
rect_main = FancyBboxPatch((1, 5), 6, 3, boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='lightblue', linewidth=2)
ax4.add_patch(rect_main)

# Label dimensions
ax4.text(4, 7.5, 'Attention Scores', fontsize=12, ha='center', fontweight='bold')
ax4.text(4, 6.8, f'[batch, heads, seq_q, seq_k]', fontsize=10, ha='center', style='italic')

# Query dimension
ax4.annotate('', xy=(0.8, 5), xytext=(0.8, 8),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax4.text(0.3, 6.5, 'Query\nDim', fontsize=10, ha='center', color='blue', fontweight='bold')

# Key dimension
ax4.annotate('', xy=(1, 4.8), xytext=(7, 4.8),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax4.text(4, 4.3, 'Key Dim (dim=-1)', fontsize=10, ha='center', color='red', fontweight='bold')

# Explanation text
explanation = """
Core Concepts:
• Each Query position decides which Key positions to "attend to"
• For each Query, attention weights over all Keys must sum to 1
• Therefore, Softmax is applied along Key dimension (dim=-1)

Example:
• Query₁ attends more to Key₀ and Key₂ → weights [0.4, 0.1, 0.4, 0.1]
• Query₂ attends more to Key₃ → weights [0.1, 0.2, 0.2, 0.5]
• Each row (each Query) has weights that sum to 1.0
"""

ax4.text(5, 2.5, explanation, fontsize=10, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('/home/ychou11/efs/ychou11/projects/Interview/API-DL/attention_softmax_visualization.png', 
            dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved to: attention_softmax_visualization.png")

plt.show()

