import os
import argparse
import numpy as np
import pandas as pd

import utils


def post_process(val):
    val_l, l1_l, l2_l = [], [], []
    for model_type in val:
        val_l.append(val[model_type][0])
        l1_l.append(val[model_type][2])
        l2_l.append(val[model_type][4])

    val_idx = val_l.index(max(val_l))
    l1_idx = l1_l.index(min(l1_l))
    l2_idx = l2_l.index(min(l2_l))

    val_tex, l1_tex, l2_tex = [], [], []
    for i, model_type in enumerate(val):
        val_tex.append(to_mean_std(val[model_type][0], val[model_type][1], best=True if i == val_idx else False))
        l1_tex.append(to_mean_std(val[model_type][2], val[model_type][3], best=True if i == l1_idx else False))
        l2_tex.append(to_mean_std(val[model_type][4], val[model_type][5], best=True if i == l2_idx else False))
    return val_tex, l1_tex, l2_tex


def to_mean_std(m, s, best):
    return "\\textbf{" + "{:.2f}".format(m) + "}" + "$\pm$ {:.2f}".format(s) if best else "{:.2f} $\pm$ {:.2f}".format(m, s)


def main(args):
    # Init dataframe
    df = pd.DataFrame(columns=['data', 'mt', 'val', 'l1', 'l2'])
    df['data'] = ['German Credit', '', '', '', '', 'SBA', '', '', '', '', 'Student Performance', '', '', '', '']
    df['mt'] = ['AR', 'MACE', 'ROAR', 'DiRRAc-NM', 'DiRRAc-GM'] * 3

    val_tex, l1_tex, l2_tex = [], [], []

    # Generate counterfactual and evaluate
    german_validity = utils.train_real_world_data('german', num_samples=args.num_samples)
    post = post_process(german_validity)
    val_tex += post[0]
    l1_tex += post[1]
    l2_tex += post[2]

    sba_validity = utils.train_real_world_data('sba', num_samples=args.num_samples)
    post = post_process(sba_validity)
    val_tex += post[0]
    l1_tex += post[1]
    l2_tex += post[2]

    student_validity = utils.train_real_world_data('student', num_samples=args.num_samples)
    post = post_process(student_validity)
    val_tex += post[0]
    l1_tex += post[1]
    l2_tex += post[2]

    df['val'] = val_tex
    df['l1'] = l1_tex
    df['l2'] = l2_tex

    # Extract csv file
    if not os.path.exists('result/real_data'):
        os.makedirs('result/real_data')
    df.to_csv(f'result/real_data/{args.save_dir}.csv', index=False)

    print("german_validity: ", german_validity)
    print("sba_validity: ", sba_validity)
    print("student_validity: ", student_validity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='validity')
    args = parser.parse_args()

    main(args)