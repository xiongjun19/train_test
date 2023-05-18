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
            'algbw': [],
            'busbw': [],
            }
    for f in f_arr:
        if f.endswith(".txt"):
            alg_bw, bus_bw = parse(os.path.join(in_dir, f))
            conn = f.strip(".txt")
            res['connect_cards'].append(conn)
            res['algbw'].append(alg_bw)
            res['busbw'].append(bus_bw)
    return res


def save_res(_dict, f_path):
    df = pd.DataFrame.from_dict(_dict)
    df.to_csv(f_path, index=False)


def parse(in_file):
    begin = False
    alg_bw = 0
    bus_bw = 0
    with open(in_file) as _in:
        for line in _in:
            if 'algbw' in line:
                begin = True
            elif begin:
                tmp = parse_val(line)
                if tmp is None:
                    continue
                alg_bw, bus_bw = tmp
    return alg_bw, bus_bw


def parse_val(line):
    line = line.strip()
    line_arr = line.split()
    if len(line_arr) > 8:
        algbw = line_arr[6]
        bus_bw = line_arr[7]
        return algbw, bus_bw
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='the dir path of the input logs')
    parser.add_argument('-o', '--output', type=str, help='the output path of the result')
    args = parser.parse_args()
    main(args)
