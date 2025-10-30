#!/usr/bin/env python

import os
import sys
import argparse
import multiprocessing

if 'DVF_SKIP_OMP_STABILIZE' not in os.environ:
    _omp_defaults = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'GOTO_NUM_THREADS': '1',
        'KMP_AFFINITY': 'disabled',
        'KMP_BLOCKTIME': '0',
        'OMP_WAIT_POLICY': 'PASSIVE',
    }
    for _k, _v in _omp_defaults.items():
        os.environ.setdefault(_k, _v)
import numpy as np
import torch
import torch.nn as nn
from Bio.Seq import Seq
from collections import defaultdict
import gc
import re




_LOOKUP = np.zeros((256, 4), dtype=np.float32)
_LOOKUP[ord('A')] = [1, 0, 0, 0]
_LOOKUP[ord('C')] = [0, 1, 0, 0]
_LOOKUP[ord('G')] = [0, 0, 1, 0]
_LOOKUP[ord('T')] = [0, 0, 0, 1]
_LOOKUP[ord('a')] = [1, 0, 0, 0]
_LOOKUP[ord('c')] = [0, 1, 0, 0]
_LOOKUP[ord('g')] = [0, 0, 1, 0]
_LOOKUP[ord('t')] = [0, 0, 0, 1]

_RC_TRANS = str.maketrans('ACGTacgt', 'TGCAtgca')


class SiameseNetwork(nn.Module):
    def __init__(self, filter_len1=10, nb_filter1=1000, nb_dense=1000, dropout_pool=0.1, dropout_dense=0.1):
        super(SiameseNetwork, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=4,
            out_channels=nb_filter1,
            kernel_size=filter_len1,
            padding='valid'
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_pool = nn.Dropout(dropout_pool)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.dense1 = nn.Linear(nb_filter1, nb_dense)
        self.dense2 = nn.Linear(nb_dense, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward_branch(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = self.global_max_pool(x)
        x = x.squeeze(-1)
        x = self.dropout_pool(x)
        x = self.relu(self.dense1(x))
        x = self.dropout_dense(x)
        x = self.sigmoid(self.dense2(x))
        return x

    def forward(self, forward_input, reverse_input):
        forward_output = self.forward_branch(forward_input)
        reverse_output = self.forward_branch(reverse_input)
        output = (forward_output + reverse_output) / 2
        return output


def parse_params_from_filename(filename):
    fl_match = re.search(r'_fl(\d+)', filename)
    fn_match = re.search(r'_fn(\d+)', filename)
    dn_match = re.search(r'_dn(\d+)', filename)
    if not (fl_match and fn_match and dn_match):
        raise ValueError(f"Cannot parse parameters from filename {filename}")
    filter_len1 = int(fl_match.group(1))
    nb_filter1 = int(fn_match.group(1))
    nb_dense = int(dn_match.group(1))
    return filter_len1, nb_filter1, nb_dense


def encode_seq_vectorized(seq):
    sb = seq.encode('ascii', 'replace')
    arr = np.frombuffer(sb, dtype=np.uint8)
    encoded = _LOOKUP[arr]

    zero_rows = np.all(encoded == 0.0, axis=1)
    if np.any(zero_rows):
        encoded[zero_rows] = 0.25

    return encoded.copy()


def encode_seq_pair_fast(seq):
    codefw = encode_seq_vectorized(seq)
    seqR = seq.translate(_RC_TRANS)[::-1]
    codebw = encode_seq_vectorized(seqR)
    return codefw, codebw


global_modDict = None
global_nullDict = None
global_device = None


def init_worker_affinity(core_num_total):
    try:
        cpu_count = os.cpu_count() or 1
        usable_cores = min(core_num_total, cpu_count)
        core_id = os.getpid() % max(1, usable_cores)
        if hasattr(os, 'sched_setaffinity'):
            try:
                os.sched_setaffinity(0, {core_id})
            except Exception:
                pass
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def load_models_parent(modDir):
    global global_modDict, global_nullDict, global_device
    global_device = torch.device('cpu')
    global_modDict = {}
    global_nullDict = {}

    for contigLengthk in ['0.15', '0.3', '0.5', '1']:
        modPattern = f'model_siamese_varlen_{contigLengthk}k'
        pth_files = [x for x in os.listdir(modDir) if modPattern in x and x.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No model found for pattern {modPattern}")
        modName = pth_files[0]
        filter_len1, nb_filter1, nb_dense = parse_params_from_filename(modName)
        model = SiameseNetwork(filter_len1, nb_filter1, nb_dense)
        checkpoint = torch.load(os.path.join(modDir, modName), map_location=global_device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(global_device)
        model.eval()
        global_modDict[contigLengthk] = model

        Y_pred_files = [x for x in os.listdir(modDir) if modPattern in x and 'Y_pred' in x]
        Y_true_files = [x for x in os.listdir(modDir) if modPattern in x and 'Y_true' in x]
        if Y_pred_files and Y_true_files:
            Y_pred_file = Y_pred_files[0]
            Y_true_file = Y_true_files[0]
            with open(os.path.join(modDir, Y_pred_file)) as f:
                Y_pred = [float(x) for x in f.read().split()]
            with open(os.path.join(modDir, Y_true_file)) as f:
                Y_true = [float(x) for x in f.read().split()]
            if 1 in Y_true:
                first_one_index = Y_true.index(1)
                arr = np.array(Y_pred[:first_one_index], dtype=np.float32)
            else:
                arr = np.array(Y_pred, dtype=np.float32)
            if arr.size:
                arr.sort()
            global_nullDict[contigLengthk] = arr
        else:
            global_nullDict[contigLengthk] = np.array([], dtype=np.float32)



def cpu_predict_worker(args):
    try:
        head, seq = args
        seq_len = len(seq)
        if seq_len < 300:
            model_key = '0.15'
        elif seq_len < 500:
            model_key = '0.3'
        elif seq_len < 1000:
            model_key = '0.5'
        else:
            model_key = '1'
        model = global_modDict[model_key]
        null_dist = global_nullDict[model_key]

        codefw, codebw = encode_seq_pair_fast(seq)
        input_fw = torch.from_numpy(np.ascontiguousarray(codefw)).unsqueeze(0).to(global_device)
        input_bw = torch.from_numpy(np.ascontiguousarray(codebw)).unsqueeze(0).to(global_device)

        with torch.no_grad():
            score_tensor = model(input_fw, input_bw)
            score = float(score_tensor.item())

        if null_dist.size:
            idx = np.searchsorted(null_dist, score, side='right')
            pvalue = float((null_dist.size - idx) / null_dist.size)
        else:
            pvalue = 1.0

        return head, seq_len, score, pvalue
    except Exception:
        return head, 0, 0.0, 1.0


class GPUPredictor:
    def __init__(self, modDir, use_amp=False, batch_gpu_chunk=64):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        self.device = torch.device('cuda')
        self.use_amp = use_amp
        self.models = {}
        self.null_distributions = {}
        self.batch_gpu_chunk = batch_gpu_chunk
        self.stream = torch.cuda.Stream()

        print('Loading models to GPU...')
        for contigLengthk in ['0.15', '0.3', '0.5', '1']:
            modPattern = f'model_siamese_varlen_{contigLengthk}k'
            pth_files = [x for x in os.listdir(modDir) if modPattern in x and x.endswith('.pth')]
            if not pth_files:
                raise FileNotFoundError(f'No model found for pattern {modPattern}')
            modName = pth_files[0]
            filter_len1, nb_filter1, nb_dense = parse_params_from_filename(modName)
            model = SiameseNetwork(filter_len1, nb_filter1, nb_dense)
            checkpoint = torch.load(os.path.join(modDir, modName), map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            self.models[contigLengthk] = model

            Y_pred_files = [x for x in os.listdir(modDir) if modPattern in x and 'Y_pred' in x]
            Y_true_files = [x for x in os.listdir(modDir) if modPattern in x and 'Y_true' in x]
            if Y_pred_files and Y_true_files:
                Y_pred_file = Y_pred_files[0]
                Y_true_file = Y_true_files[0]
                with open(os.path.join(modDir, Y_pred_file)) as f:
                    Y_pred = [float(x) for x in f.read().split()]
                with open(os.path.join(modDir, Y_true_file)) as f:
                    Y_true = [float(x) for x in f.read().split()]
                if 1 in Y_true:
                    first_one_index = Y_true.index(1)
                    arr = np.array(Y_pred[:first_one_index], dtype=np.float32)
                else:
                    arr = np.array(Y_pred, dtype=np.float32)
                if arr.size:
                    arr.sort()
                self.null_distributions[contigLengthk] = arr
            else:
                self.null_distributions[contigLengthk] = np.array([], dtype=np.float32)

        self._warmup_gpu()


    def _warmup_gpu(self):
        with torch.cuda.stream(self.stream):
            for length in [150, 300, 500, 1000]:
                dummy_fw = torch.randn(1, length, 4, dtype=torch.float32, device=self.device)
                dummy_bw = torch.randn(1, length, 4, dtype=torch.float32, device=self.device)
                model_key = self._get_model_key(length)
                model = self.models[model_key]
                with torch.no_grad():
                    if self.use_amp:
                        with torch.amp.autocast(device_type='cuda'):
                            _ = model(dummy_fw, dummy_bw)
                    else:
                        _ = model(dummy_fw, dummy_bw)
            torch.cuda.synchronize()

    def _get_model_key(self, seq_len):
        if seq_len < 300:
            return '0.15'
        elif seq_len < 500:
            return '0.3'
        elif seq_len < 1000:
            return '0.5'
        else:
            return '1'

    def predict_batch(self, batch_data):
        results = []
        grouped_sequences = defaultdict(list)
        for head, seq in batch_data:
            seq_len = len(seq)
            model_key = self._get_model_key(seq_len)
            grouped_sequences[model_key].append((head, seq))

        with torch.cuda.stream(self.stream):
            for model_key, sequences in grouped_sequences.items():
                model = self.models[model_key]
                null_dist = self.null_distributions[model_key]

                for i in range(0, len(sequences), self.batch_gpu_chunk):
                    chunk = sequences[i:i + self.batch_gpu_chunk]
                    heads = [h for h, s in chunk]
                    seqs = [s for h, s in chunk]

                    fw_list = []
                    bw_list = []
                    for s in seqs:
                        fw, bw = encode_seq_pair_fast(s)
                        fw_list.append(fw)
                        bw_list.append(bw)

                    seq_lengths = [len(seq) for seq in seqs]
                    if len(set(seq_lengths)) == 1:
                        fw_batch = np.stack([np.ascontiguousarray(x) for x in fw_list], axis=0)
                        bw_batch = np.stack([np.ascontiguousarray(x) for x in bw_list], axis=0)

                        input_fw = torch.from_numpy(fw_batch).to(self.device)
                        input_bw = torch.from_numpy(bw_batch).to(self.device)

                        with torch.no_grad():
                            if self.use_amp:
                                with torch.amp.autocast(device_type='cuda'):
                                    score_tensors = model(input_fw, input_bw)
                            else:
                                score_tensors = model(input_fw, input_bw)

                        scores = score_tensors.detach().cpu().numpy().ravel()

                        if null_dist.size:
                            idxs = np.searchsorted(null_dist, scores, side='right')
                            pvals = (null_dist.size - idxs) / null_dist.size
                        else:
                            pvals = np.ones_like(scores, dtype=np.float32)

                        for h, s_len, sc, pv in zip(heads, seq_lengths, scores, pvals):
                            results.append((h, s_len, float(sc), float(pv)))

                        del input_fw, input_bw, score_tensors
                    else:
                        for j, (h, s) in enumerate(zip(heads, seqs)):
                            fw = fw_list[j]
                            bw = bw_list[j]

                            input_fw = torch.from_numpy(np.ascontiguousarray(fw)).unsqueeze(0).to(self.device)
                            input_bw = torch.from_numpy(np.ascontiguousarray(bw)).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                if self.use_amp:
                                    with torch.amp.autocast(device_type='cuda'):
                                        score_tensor = model(input_fw, input_bw)
                                else:
                                    score_tensor = model(input_fw, input_bw)

                            score = float(score_tensor.item())

                            if null_dist.size:
                                idx = np.searchsorted(null_dist, score, side='right')
                                pvalue = float((null_dist.size - idx) / null_dist.size)
                            else:
                                pvalue = 1.0

                            results.append((h, len(s), score, pvalue))

                            del input_fw, input_bw, score_tensor

                    torch.cuda.empty_cache()

        torch.cuda.synchronize()
        return results


class FastaStreamReader:
    def __init__(self, filename, cutoff_len=1, max_n_ratio=0.3):
        self.filename = filename
        self.cutoff_len = cutoff_len
        self.max_n_ratio = max_n_ratio

    def __iter__(self):
        current_header = ""
        current_sequence = []
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_header and current_sequence:
                        seq = ''.join(current_sequence)
                        if self._should_process(seq):
                            yield current_header, seq
                    current_header = line[1:]
                    current_sequence = []
                else:
                    current_sequence.append(line)
            if current_header and current_sequence:
                seq = ''.join(current_sequence)
                if self._should_process(seq):
                    yield current_header, seq

    def _should_process(self, seq):
        if len(seq) < self.cutoff_len:
            return False
        n_count = seq.upper().count('N')
        if n_count / len(seq) > self.max_n_ratio:
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description='DeepVirFinder - Viral sequence prediction')
    parser.add_argument('-i', '--input', required=True, help='Input FASTA file')
    parser.add_argument('-o', '--output', default='./', help='Output directory')
    parser.add_argument('-m', '--model-dir', default='./models', help='Model directory')
    parser.add_argument('-l', '--length', type=int, default=100, help='Minimum sequence length')
    parser.add_argument('-c', '--cores', type=int, default=1, help='Number of CPU cores (CPU mode only)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(1)
    if not os.path.exists(args.model_dir):
        sys.exit(1)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    outfile = os.path.join(args.output, os.path.basename(args.input) + '_gt' + str(args.length) + 'bp_dvfpred.txt')

    with open(outfile, 'w') as f:
        f.write('\t'.join(['name', 'len', 'score', 'pvalue']) + '\n')

    device_str = 'CPU'
    if args.gpu and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            device_str = f'GPU ({device_name})'
        except Exception:
            device_str = 'GPU'
    else:
        device_str = 'CPU'

    print(f"Using CPU/GPU: running on {device_str}")
    print("1. Loading Models.")
    print(f"   model directory {os.path.abspath(args.model_dir)}")

    if args.gpu and torch.cuda.is_available():
        predictor = GPUPredictor(args.model_dir, use_amp=False)
        stream_reader = FastaStreamReader(args.input, args.length)
        print("2. Encoding and Predicting Sequences.")
        batch_data = []
        total_processed = 0
        with open(outfile, 'a') as out_f:
            for i, (header, seq) in enumerate(stream_reader):
                batch_data.append((header, seq))
                if len(batch_data) >= args.batch_size:
                    results = predictor.predict_batch(batch_data)
                    for head, seq_len, score, pvalue in results:
                        out_f.write('\t'.join([head, str(seq_len), f"{score:.17f}", f"{pvalue:.16f}"]) + '\n')
                    total_processed += len(batch_data)
                    print(f"Processed {total_processed} sequences...")
                    batch_data = []
                    torch.cuda.empty_cache()
            if batch_data:
                results = predictor.predict_batch(batch_data)
                for head, seq_len, score, pvalue in results:
                    out_f.write('\t'.join([head, str(seq_len), f"{score:.17f}", f"{pvalue:.16f}"]) + '\n')
                total_processed += len(batch_data)
        print("3. Done. Thank you for using DeepVirFinder.")
        print(f"   output in {os.path.abspath(outfile)}")
    else:
        try:
            ctx = multiprocessing.get_context('fork')
        except Exception:
            ctx = multiprocessing
        load_models_parent(args.model_dir)
        stream_reader = FastaStreamReader(args.input, args.length)
        print("2. Encoding and Predicting Sequences.")
        batch_data = []
        total_processed = 0
        with ctx.Pool(processes=args.cores, initializer=init_worker_affinity, initargs=(args.cores,)) as pool:
            with open(outfile, 'a') as out_f:
                for i, (header, seq) in enumerate(stream_reader):
                    batch_data.append((header, seq))
                    if len(batch_data) >= args.batch_size:
                        results_iter = pool.imap_unordered(cpu_predict_worker, batch_data)
                        for result in results_iter:
                            if result:
                                head, seq_len, score, pvalue = result
                                out_f.write('\t'.join([head, str(seq_len), f"{score:.17f}", f"{pvalue:.16f}"]) + '\n')
                        total_processed += len(batch_data)
                        print(f"Processed {total_processed} sequences...")
                        batch_data = []
                        gc.collect()
                if batch_data:
                    results_iter = pool.imap_unordered(cpu_predict_worker, batch_data)
                    for result in results_iter:
                        if result:
                            head, seq_len, score, pvalue = result
                            out_f.write('\t'.join([head, str(seq_len), f"{score:.17f}", f"{pvalue:.16f}"]) + '\n')
                    total_processed += len(batch_data)
        print("3. Done. Thank you for using DeepVirFinder.")
        print(f"   output in {os.path.abspath(outfile)}")


if __name__ == '__main__':
    main()
