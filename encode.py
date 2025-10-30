#!/usr/bin/env python
import os
import sys
import numpy as np
import optparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio.Seq import Seq

def encodeSeq(seq):
    code = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base == 'A':
            code[i, :] = [1, 0, 0, 0]
        elif base == 'T':
            code[i, :] = [0, 0, 0, 1]
        elif base == 'C':
            code[i, :] = [0, 1, 0, 0]
        elif base == 'G':
            code[i, :] = [0, 0, 1, 0]
        else:
            code[i, :] = [0.25, 0.25, 0.25, 0.25]
    return code

def process_fragment_worker(fragment, header, pos, contig_type, ncbi_name, contig_length_k):
    fw_code = encodeSeq(fragment)
    rc_fragment = str(Seq(fragment).reverse_complement())
    bw_code = encodeSeq(rc_fragment)
    header_part = header.split('/')[-1][1:].replace(' ', '_')
    contig_name = f"{contig_type}#{ncbi_name}#{contig_length_k}k#{header_part}#{pos}"
    return (f">{contig_name}", fragment, fw_code, bw_code)

def submit_fragments(executor, seq, header, contig_length, contig_type, ncbi_name, contig_length_k):
    futures = []
    pos = 0
    while pos + contig_length <= len(seq):
        fragment = seq[pos:pos+contig_length]
        if fragment.count('N') / contig_length <= 0.3:
            future = executor.submit(process_fragment_worker,
                                     fragment, header, pos,
                                     contig_type, ncbi_name, contig_length_k)
            futures.append(future)
        pos += contig_length
    return futures

class BatchStreamWriter:
    def __init__(self, out_dir, contig_type, ncbi_name, contig_length, threshold=2000000, progress_step=10000):
        self.out_dir = out_dir
        self.contig_type = contig_type
        self.ncbi_name = ncbi_name
        self.contig_length_str = f"{contig_length/1000:.2g}"
        self.threshold = threshold
        self.fileCount = 0
        self.codes_fw = []
        self.codes_bw = []
        self.seqnames = []
        self.total_fragments = 0
        self.progress_step = max(1, int(progress_step))

    def add_result(self, header, frag, fw_code, bw_code):
        self.codes_fw.append(fw_code)
        self.codes_bw.append(bw_code)
        self.seqnames.append(header)
        self.seqnames.append(frag)
        self.total_fragments += 1
        if self.total_fragments % self.progress_step == 0:
            print(f"Processed fragments: {self.total_fragments}")
        if len(self.seqnames) >= self.threshold * 2:
            self.flush_batch()

    def flush_batch(self):
        if not self.seqnames:
            return
        self.fileCount += 1
        fragmentCount = len(self.codes_fw)
        codeFileNamefw = f"{self.contig_type}#{self.ncbi_name}#{self.contig_length_str}k_num{self.fileCount}_seq{fragmentCount}_codefw.npy"
        codeFileNamebw = f"{self.contig_type}#{self.ncbi_name}#{self.contig_length_str}k_num{self.fileCount}_seq{fragmentCount}_codebw.npy"
        fastaCount = int(len(self.seqnames) / 2)
        nameFileName = f"{self.contig_type}#{self.ncbi_name}#{self.contig_length_str}k_num{self.fileCount}_seq{fastaCount}.fasta"
        print("Encoded sequences are saved in:")
        print(f"  - {codeFileNamefw}")
        print(f"  - {codeFileNamebw}")
        print(f"  - {nameFileName}")
        np.save(os.path.join(self.out_dir, codeFileNamefw), np.array(self.codes_fw), allow_pickle=False)
        np.save(os.path.join(self.out_dir, codeFileNamebw), np.array(self.codes_bw), allow_pickle=False)
        with open(os.path.join(self.out_dir, nameFileName), "w") as f:
            f.write("\n".join(self.seqnames) + "\n")
        self.codes_fw = []
        self.codes_bw = []
        self.seqnames = []

    def close(self):
        if self.seqnames:
            self.flush_batch()
        print(f"Total fragments processed: {self.total_fragments}")

def process_file(input_file, contig_length, contig_type, out_dir, num_threads, flush_threshold, progress_step):
    ncbi_name = os.path.splitext(os.path.basename(input_file))[0]
    writer = BatchStreamWriter(out_dir, contig_type, ncbi_name, contig_length, threshold=flush_threshold, progress_step=progress_step)
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        current_header = ""
        current_seq = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_header:
                        seq_str = ''.join(current_seq)
                        contig_length_k_str = f"{contig_length/1000:.2g}"
                        planned = len(seq_str) // contig_length
                        futures = submit_fragments(executor, seq_str,
                                                   current_header,
                                                   contig_length,
                                                   contig_type,
                                                   ncbi_name,
                                                   contig_length_k_str)
                        for future in as_completed(futures):
                            try:
                                header_out, frag, fw_code, bw_code = future.result()
                                writer.add_result(header_out, frag, fw_code, bw_code)
                            except Exception as e:
                                print(f"Error processing a fragment: {e}", file=sys.stderr)
                        current_seq = []
                    current_header = line
                else:
                    current_seq.append(line)
            if current_header:
                seq_str = ''.join(current_seq)
                contig_length_k_str = f"{contig_length/1000:.2g}"
                planned = len(seq_str) // contig_length
                futures = submit_fragments(executor, seq_str,
                                           current_header,
                                           contig_length,
                                           contig_type,
                                           ncbi_name,
                                           contig_length_k_str)
                for future in as_completed(futures):
                    try:
                        header_out, frag, fw_code, bw_code = future.result()
                        writer.add_result(header_out, frag, fw_code, bw_code)
                    except Exception as e:
                        print(f"Error processing a fragment: {e}", file=sys.stderr)
    writer.close()

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-i", "--input", dest="input", help="Input FASTA file")
    parser.add_option("-l", "--length", type="int", dest="length", help="Fragment length")
    parser.add_option("-p", "--type", dest="type", help="Contig type (virus/host)")
    parser.add_option("-t", "--threads", type="int", dest="threads", default=os.cpu_count(),
                      help="Number of CPU threads to use (default: all available cores)")
    (options, args) = parser.parse_args()
    
    if not all([options.input, options.length, options.type]):
        parser.print_help()
        sys.exit(1)
    
    out_dir = os.path.join(os.path.dirname(options.input), "encoded")
    os.makedirs(out_dir, exist_ok=True)

    # use fixed defaults instead of exposing CLI options
    flush_default = 2000000
    progress_default = 10000
    
    process_file(options.input, options.length, options.type, out_dir, options.threads, flush_default, progress_default)