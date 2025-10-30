import pandas as pd
from Bio import SeqIO
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--pred_file', required=True)
parser.add_argument('-f', '--fasta_file', required=True)
parser.add_argument('-o', '--output_dir', required=True)
parser.add_argument('-s', '--score', type=float, default=0.7)
parser.add_argument('-p', '--pvalue', type=float, default=0.05)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print('Reading prediction file...')

def check_name_format(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        return False
    first_data_line = lines[1].strip()
    parts = first_data_line.split('\t')
    if len(parts) >= 1:
        name_field = parts[0]
        return ' ' in name_field
    return False

def clean_and_parse(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ids = []
    scores = []
    pvalues = []
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                name_part = parts[0]
                seq_id = name_part.split()[0] if name_part else ""
                try:
                    length = int(float(parts[1]))
                except:
                    length = 0
                try:
                    score = float(parts[2])
                except:
                    score = 0.0
                try:
                    pvalue = float(parts[3])
                except:
                    pvalue = 1.0
                if seq_id and length > 0:
                    ids.append(seq_id)
                    scores.append(score)
                    pvalues.append(pvalue)
    return {'ids': ids, 'scores': scores, 'pvalues': pvalues}

if check_name_format(args.pred_file):
    data_dict = clean_and_parse(args.pred_file)
else:
    try:
        pred = pd.read_csv(args.pred_file, sep='\t')
        if 'name' not in pred.columns or 'score' not in pred.columns or 'pvalue' not in pred.columns:
            print("Warning: Column names not as expected, attempting to infer...")
            data_dict = clean_and_parse(args.pred_file)
        else:
            pred['id'] = pred['name'].str.split().str[0]
            pred = pred.dropna(subset=['score', 'pvalue'])
            data_dict = {
                'ids': pred['id'].tolist(),
                'scores': pred['score'].tolist(),
                'pvalues': pred['pvalue'].tolist()
            }
    except Exception as e:
        print(f"Standard parsing failed: {e}")
        print("Falling back to manual parsing...")
        data_dict = clean_and_parse(args.pred_file)

if len(data_dict['ids']) == 0:
    print("Error: No valid sequences found in prediction file")
    exit(1)

filtered_ids = []
for i, (seq_id, score, pvalue) in enumerate(zip(data_dict['ids'], data_dict['scores'], data_dict['pvalues'])):
    if score > args.score and pvalue < args.pvalue:
        filtered_ids.append(seq_id)

ids = set(filtered_ids)

n_total = len(data_dict['ids'])
n_pass = len(ids)
n_fail = n_total - n_pass
percent = n_pass / n_total * 100 if n_total > 0 else 0

print(f"Total: {n_total}, Passed: {n_pass}, Not passed: {n_fail}, Percent: {percent:.2f}%")

input_basename = os.path.splitext(os.path.basename(args.fasta_file))[0]
output_fasta = os.path.join(args.output_dir, f'{input_basename}_filtered.fasta')
output_ids = os.path.join(args.output_dir, f'{input_basename}_filtered_ids.txt')

print('Saving filtered IDs...')
with open(output_ids, 'w') as f:
    for i in ids:
        f.write(i + '\n')

print('Filtering fasta...')
with open(output_fasta, 'w') as out_f:
    for record in SeqIO.parse(args.fasta_file, 'fasta'):
        if record.id in ids:
            SeqIO.write(record, out_f, 'fasta')

print(f"Filtered {n_pass} sequences. Output folder: {args.output_dir}")