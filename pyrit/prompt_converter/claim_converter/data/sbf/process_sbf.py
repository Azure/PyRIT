import argparse

import pandas as pd
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

def process_sbf(input_file, data_source, model_type):
    """
    Process data by ranking models by the similarities of their annotations
    """
    data = pd.read_csv(input_file)
    data = data.dropna(subset="targetStereotype")
    data = data.loc[data.targetStereotype.str.lower() != "trivializes harm to victims"] # special case

    if data_source is not None:
        data = data.loc[data.dataSource==data_source]

    worker_counts = (
        data.groupby("HITId", as_index=False)["WorkerId"]
            .nunique().rename({"WorkerId": "n_workers"}, axis="columns")
    )
    data = data.merge(worker_counts, how="left")
    data = data.loc[data.n_workers >= 2].copy()

    st = SentenceTransformer(model_type)
    stereotype_embeds = st.encode(data.targetStereotype.tolist(), show_progress_bar=True)
    data["rowid"] = list(range(len(data)))

    distances = (
        data.groupby(["HITId", "post"], as_index=False)
            .agg({
                "rowid": lambda x: pdist(stereotype_embeds[x], "cosine").mean(),
                "targetStereotype": lambda x: " || ".join(x)
            })
            .rename({"rowid": "dists"}, axis="columns")
            .sort_values("dists")
    )
    return distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="SBIC.v2.trn.csv")
    parser.add_argument("--model_type", default="all-mpnet-base-v2")
    parser.add_argument("--data_source")
    parser.add_argument("--output_file", default="output.csv")

    args = parser.parse_args()
    distances = process_sbf(args.input_file, args.data_source, args.model_type)
    distances.to_csv(args.output_file, index=False)
