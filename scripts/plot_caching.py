import matplotlib.pyplot as plt
import numpy as np
import os

# Create data directory for plots if it doesn't exist
os.makedirs("data", exist_ok=True)

# Styling
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#999",
    "ytick.color":      "#999",
    "text.color":       "#ccc",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
})

# 1. Plot Baseline vs Caching (TTFT and Tokens)
labels = ['Baseline (No Cache)', 'Prefix Caching']
ttft_p95 = [85.3, 43.5]
tokens = [176.7, 93.0] # in thousands

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))

color = '#ff7043'
rects1 = ax1.bar(x - width/2, ttft_p95, width, label='TTFT p95 (s)', color=color)
ax1.set_ylabel('TTFT p95 (seconds)', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#4fc3f7'
rects2 = ax2.bar(x + width/2, tokens, width, label='Tokens Processed (Thousands)', color=color)
ax2.set_ylabel('Tokens (Thousands)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title('Impact of Prefix Caching (N=500)')
fig.tight_layout()
plt.savefig('data/caching_impact.png', dpi=300)
plt.close()

# 2. Plot Eviction Policies under memory pressure
policies = ['LRU', 'Cost-Ratio', 'GDS']
hit_rates = [82.78, 82.22, 83.33]
latencies = [43.5, 46.9, 43.5] # TTFT p95

x = np.arange(len(policies))

fig, ax1 = plt.subplots(figsize=(8, 5))

color = '#81c995'
ax1.bar(x - width/2, hit_rates, width, label='Hit Rate (%)', color=color)
ax1.set_ylabel('Cache Hit Rate (%)', color=color)
ax1.set_ylim([80, 85])
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#ff7043'
ax2.bar(x + width/2, latencies, width, label='TTFT p95 (s)', color=color)
ax2.set_ylabel('TTFT p95 (seconds)', color=color)
ax2.set_ylim([40, 50])
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(x)
ax1.set_xticklabels(policies)
ax1.set_title('Eviction Policy Performance (KV Blocks = 100)')
fig.tight_layout()
plt.savefig('data/eviction_policies.png', dpi=300)
plt.close()

print("Plots saved to data/caching_impact.png and data/eviction_policies.png")
