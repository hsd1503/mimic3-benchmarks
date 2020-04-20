from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse


def move_to_partition(args, patients):
    if not os.path.exists(args.dest_path):
        os.mkdir(args.dest_path)
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.dest_path, patient)
        shutil.copytree(src, dest)


def main():
    parser = argparse.ArgumentParser(description='Split a small cohort.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    parser.add_argument('dest_path', type=str, help='Directory saving files.')
    args, _ = parser.parse_known_args()

    small_set = set()
    with open(os.path.join(os.path.dirname(__file__), '../resources/smallset.csv'), "r") as small_set_file:
        for line in small_set_file:
            x = line.strip()
            small_set.add(x)

    folders = os.listdir(args.subjects_root_path)
    folders = list((filter(str.isdigit, folders)))
    samll_patients = [x for x in folders if x in small_set]

    move_to_partition(args, samll_patients)

if __name__ == '__main__':
    main()
