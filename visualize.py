import argparse
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='DeepVirFinder result visualization')
parser.add_argument('-j', '--pred_file', required=True, help='DVF prediction result TSV file')
parser.add_argument('-f', '--input_fasta', required=True, help='Original input FASTA file')
parser.add_argument('-s', '--score', type=float, default=0.7, help='Score threshold, default 0.7')
parser.add_argument('-p', '--pvalue', type=float, default=0.05, help='P-value threshold, default 0.05')
parser.add_argument('-o', '--output_dir', help='Output directory for result image (default: same as pred_file)')
args = parser.parse_args()

# Read prediction result
df = pd.read_csv(args.pred_file, sep='\t')
# Count total sequences in input fasta
n_total = sum(1 for _ in SeqIO.parse(args.input_fasta, 'fasta'))
# Filter passed sequences
passed = df[(df['score'] > args.score) & (df['pvalue'] < args.pvalue)]
n_pass = passed.shape[0]
n_fail = n_total - n_pass
percent = n_pass / n_total * 100 if n_total > 0 else 0

plt.figure(figsize=(14, 10))

# ---------- 字体大小设置（仅用于标题与坐标与编号） ----------
title_fs = 16       # 标题字号（保持之前的放大设置）
label_fs = 14       # 坐标轴标签字号（保持之前的放大设置）
annot_fs = 14       # 子图左上角编号字号（增大，如你所要求）
title_pad = 12      # 标题与图之间的距离（保持之前的设置）
# -------------------------------------------------------

# 1. Score distribution
plt.subplot(2, 2, 1)
bins = np.linspace(df['score'].min(), df['score'].max(), 30)
plt.hist(df['score'], bins=bins, color='lightgray', edgecolor='black', alpha=0.7, label='All scores')
plt.hist(df[df['score'] < 0.1]['score'], bins=bins, color='blue', alpha=0.7, label='score < 0.1')
plt.hist(df[df['score'] > 0.9]['score'], bins=bins, color='red', alpha=0.7, label='score > 0.9')
plt.axvline(0.1, color='purple', linestyle='--', linewidth=1, label='score=0.1')
plt.axvline(0.9, color='red', linestyle='--', linewidth=1, label='score=0.9')
plt.title('Score Distribution', fontsize=title_fs, pad=title_pad)
plt.xlabel('score', fontsize=label_fs)
plt.ylabel('Count', fontsize=label_fs)
plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1))
# 子图编号 (A)
plt.text(-0.05, 1.08, '(A)', transform=plt.gca().transAxes, fontsize=annot_fs, fontweight='normal', va='top', ha='left')

# 2. p-value distribution
plt.subplot(2, 2, 2)
bins_p = np.linspace(df['pvalue'].min(), df['pvalue'].max(), 30)
plt.hist(df['pvalue'], bins=bins_p, color='orange', edgecolor='black', alpha=0.7, label='All $p$-values')
plt.hist(df[df['pvalue'] < 0.05]['pvalue'], bins=bins_p, color='green', alpha=0.7, label='$p$-value < 0.05')
plt.axvline(0.05, color='red', linestyle='--', label=r'$p$=0.05')
plt.title(' $p$-value Distribution', fontsize=title_fs, pad=title_pad)
plt.xlabel(r'$p$-value', fontsize=label_fs)
plt.ylabel('Count', fontsize=label_fs)
plt.legend()
# 子图编号 (B)
plt.text(-0.05, 1.08, '(B)', transform=plt.gca().transAxes, fontsize=annot_fs, fontweight='normal', va='top', ha='left')

# Check abnormal scores
score_max = df['score'].max()
score_min = df['score'].min()
abnormal_scores = df[df['score'] > 1]
if not abnormal_scores.empty:
    print('Warning: Found score > 1!')
    print(abnormal_scores[['name', 'score', 'pvalue']])

# 3. Pie chart: Passed vs Not passed
plt.subplot(2, 2, 3)
sizes = [n_pass, n_fail]
labels = [f'Passed (score>{args.score} & $p$<{args.pvalue})', 'Not passed']
colors = ['#1976d2', '#b0bec5']
explode = (0.08, 0)
def autopct_format(pct):
    return ('%.1f%%' % pct) if pct > 0 else ''
wedges, texts, autotexts = plt.pie(
    sizes, labels=labels, colors=colors, autopct=autopct_format,
    startangle=90, counterclock=False, explode=explode, textprops={'fontsize': 12}
)
for autotext in autotexts:
    autotext.set_color('white')
plt.title('Proportion of Passed vs Not Passed', fontsize=title_fs, pad=title_pad)
plt.gca().set_aspect('equal')
# 子图编号 (C)
plt.text(-0.23, 1.08, '(C)', transform=plt.gca().transAxes, fontsize=annot_fs, fontweight='normal', va='top', ha='left')

# 4. Joint distribution (2D KDE)
plt.subplot(2, 2, 4)
# Use KDE with more compatible parameters
try:
    # Create KDE plot with proper subplot context
    ax4 = plt.gca()
    sns.kdeplot(
        data=df, x='score', y='pvalue',
        fill=True, cmap='Blues', thresh=0.1, levels=20, alpha=0.6,
        ax=ax4
    )
    from matplotlib.patches import Patch
    kde_patch = Patch(color='lightblue', alpha=0.6, label='KDE density')
except Exception as e:
    plt.hexbin(df['score'], df['pvalue'], gridsize=20, cmap='Blues', alpha=0.8)
    print(f"KDE plot failed, using hexbin instead: {e}")
    from matplotlib.patches import Patch
    kde_patch = Patch(color='lightblue', alpha=0.8, label='Hexbin density')

plt.scatter(df['score'], df['pvalue'], s=10, color='gray', alpha=0.3, label='Samples')
plt.axvline(0.9, color='red', linestyle='--', linewidth=1, label='score=0.9')
plt.axhline(0.05, color='green', linestyle='--', linewidth=1, label=r'$p$-value=0.05')
plt.title('Joint Distribution of Score and $p$-value (KDE)', fontsize=title_fs, pad=title_pad)
plt.xlabel('score', fontsize=label_fs)
plt.ylabel(r'$p$-value', fontsize=label_fs)
plt.xlim(0, 1)
plt.legend()
# 子图编号 (D)
plt.text(-0.05, 1.08, '(D)', transform=plt.gca().transAxes, fontsize=annot_fs, fontweight='normal', va='top', ha='left')

# ========== 仅增加：上下子图间距和左右间距，避免编号与内容重叠 ==========
plt.subplots_adjust(hspace=0.5, wspace=0.45)
# =========================================================

plt.tight_layout()
# Save output to specified directory or same directory as pred_file
if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
    input_prefix = os.path.splitext(os.path.basename(args.pred_file))[0]
    out_path = os.path.join(args.output_dir, f'{input_prefix}_visualization.png')
else:
    output_dir = os.path.dirname(os.path.abspath(args.pred_file))
    input_prefix = os.path.splitext(os.path.basename(args.pred_file))[0]
    out_path = os.path.join(output_dir, f'{input_prefix}_visualization.png')
plt.savefig(out_path, dpi=300)
plt.show()
print(f'Done. Result image saved to: {out_path}')
