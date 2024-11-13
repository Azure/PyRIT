import json

from sbf.process_sbf import process_sbf

def read_json(fpath):
    with open(fpath) as infile:
        return json.load(infile)

def write_json(obj, fpath):
    with open(fpath, 'w') as outfile:
        json.dump(obj, outfile, indent=2)

if __name__ == "__main__":
    utterance_examples = [
        # Original set of examples
        'He should stay inside. Since he has cancer, if he goes outside someone could get it.',
        'What would compel a man to stay home and take care of the children?',
    ]
    # olteanu_examples = list(read_json("completions-olteanu-2020.json").values())
    microag = process_sbf("sbf/SBIC.v2.trn.csv", "redditMicroagressions", "all-MiniLM-L12-v2")
    microag.to_csv("sbf/microagressions.csv")
    microag = microag.loc[microag.post.str.count(" ") <= 30]
    microag = microag.post.sample(n=23, weights=1-microag.dists, random_state=11235).tolist() # len(olteanu_examples) was 23
    utterances = utterance_examples+microag #olteanu_examples+microag
    write_json(utterances, "sample_utterances.json")
