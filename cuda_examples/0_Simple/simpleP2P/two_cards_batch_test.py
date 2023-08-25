# coding=utf8

import argparse
import os
import subprocess
from tqdm import tqdm



def main(args):
    card_num =  args.card_num
    pair_arr = _gen_arr(card_num)
    d = dict(os.environ)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    shell_cmd = './test_run'
    for pair in tqdm(pair_arr):
        d['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, pair))
        out_file = os.path.join(out_dir, "_".join(map(str, pair)) + ".txt")
        with open(out_file, "w") as _out:
            subprocess.call(shell_cmd, shell=True, stdout=_out, env=d)


def _gen_arr(num):
    res = []
    for i in range(num):
        for j in range(i+1, num):
            res.append([i,j])
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str, default='2_cards_out')
    parser.add_argument('-n', '--card_num', type=int, default=10)
    args = parser.parse_args()
    main(args)
