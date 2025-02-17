"""
Compute confidence intervals for sequences from a CSV. 
Output is written to a new CSV file, with the lower and upper bounds of the confidence intervals on alternating rows.

Usage:
    python csv_conf.py delta prior_type csv_path out_path

Arguments:
    d          - delta value, between 0 and 1
    prior_type - prior type, either 'minmax' or 'truncated'
    csv_path   - path to input CSV file
    out_path   - path to output CSV file

Example:
    python csv_conf.py 0.1 minmax data.csv outfile.csv
"""

import argparse
import csv
import numpy as np
from collections import deque
import tqdm as tqdm
import priors
import extremal_conf as ec
import betting_util as butil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('d', type=float)
    parser.add_argument('ptype', choices=['minmax', 'truncated'])
    parser.add_argument('csv_path')
    parser.add_argument('csv_out')

    args = parser.parse_args()
    
    row_arrays = deque()
    with open(args.csv_path) as f:
        for row in csv.reader(f):
            clean_row = [float(x) for x in row if x.strip().upper() != 'NA']
            if clean_row:
                row_arrays.append(np.array(clean_row))
    
    confs = deque()
    pcache = {}
    for seq in tqdm.tqdm(row_arrays):
        T = len(seq)
        if not T in pcache:
            if args.ptype == 'minmax':
                pr = butil.marginal_priors(priors.log_minmax_prior(T))
                assert(np.allclose(butil.sum_marginals(pr), 1))
                pcache[T] = pr
            elif args.ptype == 'truncated':
                pr = butil.marginal_priors(priors.truncated_prior(T, args.d))
                marg_sums = butil.sum_marginals(pr)
                assert(np.allclose(marg_sums, 1))
                pcache[T] = pr
        else:
            pr = pcache[T]
        lcb, ucb = ec.conf(seq, pr, args.d)
        confs.append((lcb, ucb))

    with open(args.csv_out, 'w',newline='') as f:
        writer = csv.writer(f)
        for lcb, ucb in confs:
            writer.writerow(lcb)
            writer.writerow(ucb)
if __name__ == "__main__":
    main()