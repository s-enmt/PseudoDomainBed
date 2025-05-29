# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed import augmentation_algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

from icecream import ic

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{ll" + "c" * (num_cols-2) + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%").replace("_", "\\_") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    # read augmentation names and sort
    aug_names = Q(records).select("args.augmentations").unique().sorted()

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset.replace("_", "\\_")))
        train_envs = range(datasets.num_environments(dataset))

        table = []
        for i, algorithm in enumerate(alg_names):
            for j, augmentation in enumerate(aug_names):
                row = [algorithm, "+".join(augmentation)]
                means = []
                for j, train_env in enumerate(train_envs):
                    test_env = str([env for env in train_envs if env != train_env])
                    trial_accs = (grouped_records
                        .filter_equals(
                            "dataset, algorithm, augmentations, test_env",
                            (dataset, algorithm, str(augmentation), test_env)
                        ).select("sweep_acc"))
                    mean, err, value = format_mean(trial_accs, latex)
                    means.append(mean)
                    row.append(value)
                # Avg
                if None in means:
                    row.append("X")
                else:
                    row.append("{:.1f}".format(sum(means) / len(means)))
                if not all(x=='X' for x in row[2:]):
                    table.append(row)
        col_labels = [
            "Algorithm", 
            "Pseudo-domain",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, list(col_labels),
            colwidth=None, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = []
    for i, algorithm in enumerate(alg_names):
        for j, augmentation in enumerate(aug_names):
            row = [algorithm, "+".join(augmentation)]
            means = []
            for j, dataset in enumerate(dataset_names):
                trial_averages = (grouped_records
                    .filter_equals(
                        "algorithm, augmentations, dataset", 
                        (algorithm, str(augmentation), dataset))
                    .group("trial_seed")
                    .map(lambda trial_seed, group:
                        group.select("sweep_acc").mean()
                    )
                )
                mean, err, value = format_mean(trial_averages, latex)
                means.append(mean)
                row.append(value)
            # Avg
            if None in means:
                row.append("X")
            else:
                row.append("{:.1f}".format(sum(means) / len(means)))
            if not all(x=='X' for x in row[2:]):
                table.append(row)

    col_labels = ["Algorithm", "Pseudo-domain", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, col_labels, colwidth=None,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)
   
    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}") 
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        # model_selection.LeaveOneOutSelectionMethod,
        # model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name)) 
        print_results_tables(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
