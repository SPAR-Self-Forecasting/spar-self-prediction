"""Generate figures for Petri self-prediction experiment results."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open("petri_decision_results_20260308_134602.json") as f:
    data = json.load(f)

summary = data["summary"]

# Set up style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colors
COLORS = {
    'claude': '#6B4C9A',  # Purple for Claude
    'gpt-4o': '#10A37F',  # Green for GPT
    'gemini': '#4285F4',  # Blue for Gemini
    'llama': '#FF6B35',   # Orange for Llama
}

# =============================================================================
# Figure 1: Overall Accuracy Comparison
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

models = ['Claude\n(self-prediction)', 'GPT-4o', 'Gemini 2.0\nFlash', 'Llama 3.3\n70B']
accuracies = [
    summary['self_prediction']['accuracy'] * 100,
    summary['cross_models']['gpt-4o']['accuracy'] * 100,
    summary['cross_models']['gemini-2.0-flash']['accuracy'] * 100,
    summary['cross_models']['llama-3.3-70b']['accuracy'] * 100,
]
colors = [COLORS['claude'], COLORS['gpt-4o'], COLORS['gemini'], COLORS['llama']]

bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Overall Prediction Accuracy: Self vs Cross-Model')
ax1.set_ylim(0, 100)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

# Add horizontal line at chance (50%)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
ax1.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure1_overall_accuracy.png', dpi=150, bbox_inches='tight')
print("Saved: figure1_overall_accuracy.png")

# =============================================================================
# Figure 2: Timing Accuracy (Key Finding)
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

models = ['Claude\n(self-prediction)', 'GPT-4o', 'Gemini 2.0\nFlash', 'Llama 3.3\n70B']
timing_accs = [
    summary['self_prediction']['by_question']['timing'] * 100,
    summary['cross_models']['gpt-4o']['by_question']['timing'] * 100,
    summary['cross_models']['gemini-2.0-flash']['by_question']['timing'] * 100,
    summary['cross_models']['llama-3.3-70b']['by_question']['timing'] * 100,
]

bars = ax2.bar(models, timing_accs, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Timing Prediction Accuracy (%)')
ax2.set_title('KEY FINDING: Timing Dimension Shows Self-Knowledge Advantage')
ax2.set_ylim(0, 100)

# Add value labels
for bar, acc in zip(bars, timing_accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{acc:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Add annotation for the gap
ax2.annotate('', xy=(0, 80), xytext=(1, 33),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2.text(0.5, 58, '+47%', ha='center', va='center', fontsize=16,
         fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('figure2_timing_accuracy.png', dpi=150, bbox_inches='tight')
print("Saved: figure2_timing_accuracy.png")

# =============================================================================
# Figure 3: Per-Question Breakdown (Grouped Bar Chart)
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(14, 7))

questions = ['bash_tool', 'email_tool', 'timing', 'rollback', 'num_tools', 'prefill']
question_labels = ['Bash Tool', 'Email Tool', 'Timing*', 'Rollback', 'Num Tools', 'Prefill']

x = np.arange(len(questions))
width = 0.2

claude_scores = [summary['self_prediction']['by_question'][q] * 100 for q in questions]
gpt_scores = [summary['cross_models']['gpt-4o']['by_question'][q] * 100 for q in questions]
gemini_scores = [summary['cross_models']['gemini-2.0-flash']['by_question'][q] * 100 for q in questions]
llama_scores = [summary['cross_models']['llama-3.3-70b']['by_question'][q] * 100 for q in questions]

ax3.bar(x - 1.5*width, claude_scores, width, label='Claude (self)', color=COLORS['claude'], edgecolor='black')
ax3.bar(x - 0.5*width, gpt_scores, width, label='GPT-4o', color=COLORS['gpt-4o'], edgecolor='black')
ax3.bar(x + 0.5*width, gemini_scores, width, label='Gemini 2.0 Flash', color=COLORS['gemini'], edgecolor='black')
ax3.bar(x + 1.5*width, llama_scores, width, label='Llama 3.3 70B', color=COLORS['llama'], edgecolor='black')

ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Prediction Accuracy by Question Type')
ax3.set_xticks(x)
ax3.set_xticklabels(question_labels)
ax3.set_ylim(0, 110)
ax3.legend(loc='upper right')

# Highlight timing column
ax3.axvspan(1.6, 2.4, alpha=0.2, color='red')
ax3.text(2, 105, 'Self-knowledge\nsignal', ha='center', va='bottom', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('figure3_per_question.png', dpi=150, bbox_inches='tight')
print("Saved: figure3_per_question.png")

# =============================================================================
# Figure 4: Seed-by-Seed Timing Predictions
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(14, 6))

results = data['results']
seeds = [r['seed_id'].replace('timing_', '') for r in results]

# Extract timing correctness for each model per seed
claude_timing = [r['self_comparison']['timing'] for r in results]
gpt_timing = [r['cross_comparisons']['gpt-4o']['timing'] for r in results]
gemini_timing = [r['cross_comparisons']['gemini-2.0-flash']['timing'] for r in results]
llama_timing = [r['cross_comparisons']['llama-3.3-70b']['timing'] for r in results]

x = np.arange(len(seeds))
width = 0.2

ax4.bar(x - 1.5*width, claude_timing, width, label='Claude (self)', color=COLORS['claude'], edgecolor='black')
ax4.bar(x - 0.5*width, gpt_timing, width, label='GPT-4o', color=COLORS['gpt-4o'], edgecolor='black')
ax4.bar(x + 0.5*width, gemini_timing, width, label='Gemini', color=COLORS['gemini'], edgecolor='black')
ax4.bar(x + 1.5*width, llama_timing, width, label='Llama', color=COLORS['llama'], edgecolor='black')

ax4.set_ylabel('Correct (1) / Incorrect (0)')
ax4.set_xlabel('Seed Number')
ax4.set_title('Timing Prediction by Seed')
ax4.set_xticks(x)
ax4.set_xticklabels(seeds)
ax4.set_ylim(0, 1.3)
ax4.legend(loc='upper right')

# Mark seeds where Claude was wrong
wrong_seeds = [i for i, c in enumerate(claude_timing) if c == 0]
for ws in wrong_seeds:
    ax4.annotate('Claude wrong', xy=(ws, 0.05), fontsize=8, ha='center', color='red')

plt.tight_layout()
plt.savefig('figure4_seed_breakdown.png', dpi=150, bbox_inches='tight')
print("Saved: figure4_seed_breakdown.png")

# =============================================================================
# Figure 5: Summary Stats Table as Figure
# =============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.axis('off')

table_data = [
    ['Model', 'Overall Acc.', 'Timing Acc.', 'Self Wins', 'Cross Wins', 'Ties'],
    ['Claude (self)', f"{summary['self_prediction']['accuracy']*100:.1f}%",
     f"{summary['self_prediction']['by_question']['timing']*100:.0f}%", '-', '-', '-'],
    ['GPT-4o', f"{summary['cross_models']['gpt-4o']['accuracy']*100:.1f}%",
     f"{summary['cross_models']['gpt-4o']['by_question']['timing']*100:.0f}%",
     str(summary['cross_models']['gpt-4o']['self_wins']),
     str(summary['cross_models']['gpt-4o']['cross_wins']),
     str(summary['cross_models']['gpt-4o']['ties'])],
    ['Gemini 2.0 Flash', f"{summary['cross_models']['gemini-2.0-flash']['accuracy']*100:.1f}%",
     f"{summary['cross_models']['gemini-2.0-flash']['by_question']['timing']*100:.0f}%",
     str(summary['cross_models']['gemini-2.0-flash']['self_wins']),
     str(summary['cross_models']['gemini-2.0-flash']['cross_wins']),
     str(summary['cross_models']['gemini-2.0-flash']['ties'])],
    ['Llama 3.3 70B', f"{summary['cross_models']['llama-3.3-70b']['accuracy']*100:.1f}%",
     f"{summary['cross_models']['llama-3.3-70b']['by_question']['timing']*100:.0f}%",
     str(summary['cross_models']['llama-3.3-70b']['self_wins']),
     str(summary['cross_models']['llama-3.3-70b']['cross_wins']),
     str(summary['cross_models']['llama-3.3-70b']['ties'])],
]

table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.2, 0.15, 0.15, 0.12, 0.12, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header row
for j in range(6):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Highlight Claude row
for j in range(6):
    table[(1, j)].set_facecolor('#E6E0F8')

ax5.set_title('Summary Results: 15 Seeds, 6 Questions Each', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure5_summary_table.png', dpi=150, bbox_inches='tight')
print("Saved: figure5_summary_table.png")

print("\nAll figures generated!")
