import os
import sys
import argparse
import subprocess
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_deepvirfinder(input_fasta, output_dir, dvf_script, model_dir, cutoff_len, core_num):
    import re
    input_fasta = os.path.normpath(os.path.abspath(os.path.expanduser(re.sub(r'/+', '/', input_fasta))))
    output_dir = os.path.normpath(os.path.abspath(os.path.expanduser(output_dir)))
    model_dir = os.path.normpath(os.path.abspath(os.path.expanduser(model_dir)))
    
    output_pred = os.path.join(output_dir, os.path.basename(input_fasta) + f'_gt{cutoff_len}bp_dvfpred.txt')
    
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input file not found: {input_fasta}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, dvf_script,
        '-i', input_fasta,
        '-o', output_dir,
        '-m', model_dir,
        '-l', str(cutoff_len),
        '-c', str(core_num)
    ]
    print(f"[Step 1] Run DeepVirFinder: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: dvf.py execution failed with exit code: {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        raise
    
    return output_pred

def filter_sequences(pred_file, fasta_file, output_dir, score_thres, p_thres):
    output_fasta = os.path.join(output_dir, f'filtered_score{score_thres}_p{p_thres}.fasta')
    output_ids = os.path.join(output_dir, f'filtered_ids.txt')
    pred = pd.read_csv(pred_file, sep='\t')
    pred['id'] = pred['name'].str.split().str[0]
    filtered = pred[(pred['score'] > score_thres) & (pred['pvalue'] < p_thres)]
    ids = set(filtered['id'])

    n_total = pred.shape[0]
    n_pass = len(ids)
    n_fail = n_total - n_pass
    percent = n_pass / n_total * 100 if n_total > 0 else 0

    print(f"Total sequences: {n_total}")
    print(f"Passed filter (score>{score_thres} & pvalue<{p_thres}): {n_pass}")
    print(f"Not passed: {n_fail}")
    print(f"Percent passed: {percent:.2f}%")

    with open(output_ids, 'w') as f:
        for i in ids:
            f.write(i + '\n')

    with open(output_fasta, 'w') as out_f:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            if record.id in ids:
                SeqIO.write(record, out_f, 'fasta')

    print(f"Filtered {n_pass} sequences. Output: {output_fasta}, {output_ids}")
    return output_fasta, output_ids, n_pass, n_fail, percent

def plot_results(pred_file, filtered_ids_file, output_dir, n_pass, n_fail, percent):
    df = pd.read_csv(pred_file, sep='\t')
    with open(filtered_ids_file) as f:
        filtered_ids = set(line.strip() for line in f)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    bins = np.linspace(df['score'].min(), df['score'].max(), 30)
    plt.hist(df['score'], bins=bins, color='lightgray', edgecolor='black', alpha=0.7, label='All scores')
    plt.hist(df[df['score'] < 0.1]['score'], bins=bins, color='blue', alpha=0.7, label='score < 0.1')
    plt.hist(df[df['score'] > 0.9]['score'], bins=bins, color='red', alpha=0.7, label='score > 0.9')
    plt.axvline(0.1, color='purple', linestyle='--', linewidth=1, label='score=0.1')
    plt.axvline(0.9, color='red', linestyle='--', linewidth=1, label='score=0.9')
    plt.title('Score Distribution')
    plt.xlabel('score')
    plt.ylabel('Count')
    plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1))

    plt.subplot(2, 2, 2)
    bins_p = np.linspace(df['pvalue'].min(), df['pvalue'].max(), 30)
    plt.hist(df['pvalue'], bins=bins_p, color='orange', edgecolor='black', alpha=0.7, label='All p-values')
    plt.hist(df[df['pvalue'] < 0.05]['pvalue'], bins=bins_p, color='green', alpha=0.7, label='p-value < 0.05')
    plt.axvline(0.05, color='red', linestyle='--', label='p=0.05')
    plt.title('p-value Distribution')
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(2, 2, 3)
    sizes = [n_pass, n_fail]
    labels = [f'Passed ({percent:.1f}%)', 'Not passed']
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
    plt.title('Proportion of Passed vs Not Passed', fontsize=14)
    plt.gca().set_aspect('equal')

    plt.subplot(2, 2, 4)
    sns.kdeplot(
        x=df['score'], y=df['pvalue'],
        fill=True, cmap='Blues', thresh=0.05, levels=100, alpha=0.8
    )
    plt.scatter(df['score'], df['pvalue'], s=10, color='gray', alpha=0.3, label='Samples')
    plt.axvline(0.9, color='red', linestyle='--', linewidth=1)
    plt.axhline(0.05, color='green', linestyle='--', linewidth=1)
    plt.title('Joint Distribution of Score and p-value (KDE)')
    plt.xlabel('score')
    plt.ylabel('p-value')
    plt.xlim(0, 1)
    plt.legend(['score=0.9', 'p-value=0.05', 'Samples'])

    plt.tight_layout()
    out_png = os.path.join(output_dir, 'dvf_result_summary.png')
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"Visualization saved to: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="DeepVirFinder full pipeline (all-in-one)")
    parser.add_argument('-i', '--in', dest='input', required=True, help='Input fasta file')
    parser.add_argument('-o', '--out', dest='outdir', required=True, help='Output directory')
    parser.add_argument('-m', '--mod', dest='model_dir', default='models', help='DeepVirFinder model directory')
    parser.add_argument('-l', '--len', dest='cutoff_len', type=int, default=1500, help='Minimum sequence length, default 1500')
    parser.add_argument('-c', '--core', dest='core', type=int, default=4, help='Number of parallel cores, default 4')
    parser.add_argument('--score', type=float, default=0.7, help='Score threshold, default 0.7')
    parser.add_argument('--pvalue', type=float, default=0.05, help='P-value threshold, default 0.05')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("\n==============================")
    print('Step 1: Running DeepVirFinder')
    print("==============================")
    pred_file = run_deepvirfinder(
        args.input, args.outdir, 'dvf.py', args.model_dir, args.cutoff_len, args.core
    )

    print("\n==============================")
    print('Step 2: Filtering viral sequences')
    print("==============================")
    filtered_fasta, filtered_ids, n_pass, n_fail, percent = filter_sequences(
        pred_file, args.input, args.outdir, args.score, args.pvalue
    )
    print(f"\nSummary after filtering:")
    print(f"  Total sequences: {n_pass + n_fail}")
    print(f"  Passed filter: {n_pass}")
    print(f"  Not passed: {n_fail}")
    print(f"  Percent passed: {percent:.2f}%")
    print(f"  Filtered FASTA: {filtered_fasta}")
    print(f"  Filtered IDs: {filtered_ids}")

    print("\n==============================")
    print('Step 3: Visualizing results')
    print("==============================")
    plot_results(pred_file, filtered_ids, args.outdir, n_pass, n_fail, percent)

    print("\n==============================")
    print('Pipeline finished!')
    print(f'All results are saved in: {args.outdir}')
    print("==============================\n")

if __name__ == '__main__':
    main() 