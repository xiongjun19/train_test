# coding=utf8


import os
import argparse
import pandas as pd


def main(args):
    res = parse_batchly(args.input)
    save_res(res, args.output)


def parse_batchly(in_dir):
    f_arr = os.listdir(in_dir)
    res = {
            'connect_cards': [],
            'bandwidth': [],
            }
    for f in f_arr:
        if f.endswith(".txt"):
            bw = parse(os.path.join(in_dir, f))
            conn = f.strip(".txt")
            res['connect_cards'].append(conn)
            res['bandwidth'].append(bw)
    return res


def save_res(_dict, f_path):
    df = pd.DataFrame.from_dict(_dict)
    df.to_csv(f_path, index=False)


def parse(in_file):
    with open(in_file) as _in:
        for line in _in:
            if 'cudaMemcpy' in line:
                lin_arr = line.split(":") 
                bw = lin_arr[-1].strip()
                return bw
    return None 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='the dir path of the input logs')
    parser.add_argument('-o', '--output', type=str, help='the output path of the result')
    args = parser.parse_args()
    main(args)
