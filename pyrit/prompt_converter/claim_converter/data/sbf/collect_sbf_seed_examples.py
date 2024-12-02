# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords


def _unpack_list(elements):
    return [x for sublist in elements for x in sublist]


def _get_max_count_with_counter(elements, counter):
    return max(counter[x] for x in elements)


def _get_max_element_with_counter(elements, counter):
    return max(elements, key=lambda x: counter[x])


def collect_seed_examples(input_file, examples_per_group=2, max_sents=2, balance_sources=True, seed=11235):
    """
    Process data by ranking models by the similarities of their annotations
    """
    data = pd.read_csv(input_file)
    data = data.dropna(subset="targetStereotype")

    # exclude jokes
    data = data.loc[data.dataSource != "r/offensivequestions"]
    data = data.loc[~data.dataSource.str.contains("joke", regex=True)]

    # annotated by multiple workers
    worker_counts = (
        data.groupby("HITId", as_index=False)["WorkerId"].nunique().rename({"WorkerId": "n_workers"}, axis="columns")
    )
    data = data.merge(worker_counts, how="left")
    data = data.loc[data.n_workers >= 2].copy()
    data["rowid"] = list(range(len(data)))

    # collect frequency of targets
    data["target"] = data.targetMinority.str.lower().str.split(", ")
    target_counts = Counter([t for ts in data.target for t in ts])

    # collect frequency of stereotypes
    cv = CountVectorizer(stop_words=stopwords.words("english"), ngram_range=(1, 1))
    doc_terms = cv.fit_transform(data.targetStereotype)
    term_count_ranks = pd.Series(doc_terms.sum(0).A1).rank(method="min").values

    # aggregate annotations by example
    aggregated = data.groupby(["HITId", "post", "dataSource"], as_index=False).agg(
        stereotype=("targetStereotype", lambda stereo: " ".join(stereo)),
        target=("target", lambda targets: list(set(_unpack_list(targets)))),
        top_target=("target", lambda targets: _get_max_element_with_counter(_unpack_list(targets), target_counts)),
        target_count=("target", lambda targets: _get_max_count_with_counter(_unpack_list(targets), target_counts)),
    )
    # filter out examples with too many sentences
    aggregated["n_sents"] = [len(sent_tokenize(post)) for post in aggregated.post]
    aggregated = aggregated.loc[aggregated.n_sents <= max_sents]

    # avoid over-representation by any one source
    if balance_sources:
        min_source_size = aggregated.dataSource.value_counts().min()
        aggregated = aggregated.groupby("dataSource").sample(n=min_source_size, random_state=seed)

    # find most and least common stereotypes by target
    examples = []
    n_ex = examples_per_group // 2
    for target, group in aggregated.groupby("top_target"):
        if len(group) <= examples_per_group:
            examples.append(group)
            continue
        # get terms appearing in each annotated stereotype
        group_terms = cv.transform(group.stereotype).todense().A  # n_examples x n_terms
        # find the least common term used to describe the target in each example
        group_terms[group_terms == 0] = 1e10
        stereotypes_by_term_freq = (group_terms * term_count_ranks[None, :]).min(1).argsort()
        idxs = stereotypes_by_term_freq[:n_ex].tolist() + stereotypes_by_term_freq[-n_ex:].tolist()
        group_subset = group.iloc[idxs].copy()
        group_subset["stereotype_frequency"] = ["low"] * n_ex + ["high"] * n_ex
        examples.append(group_subset)
    examples = pd.concat(examples)
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="SBIC.v2.trn.csv")
    parser.add_argument("--max_sents", default=2, type=int)
    parser.add_argument("--examples_per_group", default=2, type=int)
    parser.add_argument("--balance_sources", default=True, action="store_true")
    parser.add_argument("--no_balance_sources", dest="balance_sources", action="store_false")
    parser.add_argument("--output_file", default="seed_examples.csv")
    args = parser.parse_args()
    nltk.download("stopwords")

    # must have even numbers of examples per group
    if args.examples_per_group % 2 != 0:
        raise ValueError(
            "`examples_per_group` must be even, since we need to collect both high and low frequency examples."
        )

    args = parser.parse_args()
    examples = collect_seed_examples(
        args.input_file,
        examples_per_group=args.examples_per_group,
        max_sents=args.max_sents,
        balance_sources=args.balance_sources,
    )
    examples.to_csv(args.output_file, index=False)
