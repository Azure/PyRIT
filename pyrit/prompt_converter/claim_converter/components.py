import re
import json
import itertools
import random
import string
from typing import List
from pathlib import Path
from datetime import datetime

import torch
import spacy
import yaml
import numpy as np
import pandas as pd

# import streamlit as st
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import wordnet
from nltk.parse.generate import generate

import transformers
from transformers import pipeline, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria

from pyrit.prompt_converter.claim_converter import prompt_openai, exemplars
from pyrit.prompt_converter.claim_converter.classifiers import ClaimClassifierCE, ClaimClassifierSF

# from prompt_openai import make_prompt, complete_prompt, _OPENAI_MAX_PROMPTS
# from exemplars import FEW_SHOT_INTERNAL

try:
    import fasttext
    import fasttext.util

    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False


try:
    import inflect

    _INFLECT_AVAILABLE = True
except ImportError:
    _INFLECT_AVAILABLE = False


##########################
# Session state management
##########################
def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _init_styling(annotate_unrelated_failures=True):
    # Styling
    pass
    # st.markdown("""
    #     <style>
    #         .streamlit-wide
    #         .appview-container
    #         .main
    #         .block-container {
    #             max-width: 70%;
    #         }

    #         textarea {
    #             font-family: "Source Code Pro", monospace !important;
    #         }
    #     </style>""",
    #     unsafe_allow_html=True
    # )

    if annotate_unrelated_failures:
        annotator_css = Path("./annotator.css").read_text()  # TODO: memoize?
        # st.markdown(f"<style>{annotator_css}</style>", unsafe_allow_html=True)


def _init_state():
    # Init states
    pass
    # if "max_completed_step" not in st.session_state:
    #     st.session_state["max_completed_step"] = 0
    # if "do_fit" not in st.session_state:
    #     st.session_state["do_fit"] = True
    # if "num_prior_annotations" not in st.session_state:
    #     st.session_state["num_prior_annotations"] = 0
    # if "session_id" not in st.session_state:
    #     st.session_state["session_id"] = datetime.now().isoformat(timespec="minutes", sep="-").replace(":", "")


# @st.cache_data(show_spinner=False)
def init_output_dir(output_dir, utterance, session_id, config):
    """
    Create a new output directory
    """
    utt_clean = re.sub(f"[{string.punctuation}]", "", utterance.lower())
    id = "-".join(utt_clean.split()[:5])
    output_dir = Path(output_dir, id, session_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    config["selected_utterance"] = utterance
    with open(output_dir / "config.yml", "w") as outfile:
        yaml.safe_dump(config, outfile)
    return output_dir


# Below logic is designed to only "advance" the streamlit app once user operations
# are complete: idea is that a user performs some actions, clicks "go", then the app
# moves to the next stage.
#
# We do this by maintaining an (ephemeral) `CURRENT_STEP` counter that
# advances as we progress through the app script, which in turn updates a persistent
# `max_completed_step` value in the `st.session_state`. The `max_completed_step` dictates
# where the app will stop when a widget is modified.
#
# This is accomplished by the `_reset_to_step()` widget callback, which will change
# when a widget is modified. When a user changes a widget upstream, then the `max_completed_step` is
# changed to the `CURRENT_STEP` where that widget exists.
#
# The app will then stop (`st.stop()`) at either the `_proceed_to_next_step` or
# `_submit_form_and_proceed_to_next_step` after the changed widget.
# These functions present a "Continue"/"Submit" button to the user which
# will advance (i.e., increment by 1) both the `max_completed_step` and `CURRENT_STEP`
def _reset_to_step(step):
    def _reset_max_step():
        pass
        # st.session_state["max_completed_step"] = step

    return _reset_max_step


# def _proceed_to_next_step(current_step, label="Continue"):
#     before_last_completed_step = current_step < st.session_state["max_completed_step"]
#     proceed = st.button(
#         label,
#         on_click=_reset_to_step(current_step+1),
#         disabled=before_last_completed_step,
#         key=f"continue_from_step_{current_step}",
#     )
#     if proceed or before_last_completed_step:
#         return current_step + 1
#     else:
#         st.stop()

# def _submit_form_and_proceed_to_next_step(current_step):
#     before_last_completed_step = current_step < st.session_state["max_completed_step"]
#     proceed = st.form_submit_button(
#         "Submit",
#         on_click=_reset_to_step(current_step+1),
#     )
#     if proceed or before_last_completed_step:
#         return current_step + 1
#     else:
#         st.stop()

####################
# Interface elements
####################
# def few_shot_editor(label, few_shot_exemplars, **kwargs):
#     """
#     Edit few-shot model examples in the interface
#     """
#     if "height" not in kwargs:
#         kwargs["height"] = len(few_shot_exemplars) * 24
#     few_shot_str = json.dumps(few_shot_exemplars, indent=2)
#     few_shot_edited = st.text_area(label, few_shot_str, **kwargs)
#     try:
#         return json.loads(few_shot_edited)
#     except json.JSONDecodeError:
#         st.error("json decoding failed. Is there a trailing comma?")
#         st.stop()


# def build_annotator_aggrid(df, pre_selected_idx=None, cols_to_display=None):
#     """
#     Annotate an interactive dataframe with AG Grid
#     """
#     # from st_aggrid import AgGrid, ColumnsAutoSizeMode
#     # from st_aggrid.grid_options_builder import GridOptionsBuilder
#     display_df = df
#     if cols_to_display is not None:
#         display_df = df[cols_to_display]
#     gd = GridOptionsBuilder.from_dataframe(
#         display_df,
#         filterable=False,
#         sortable=False,
#         groupable=False,
#     )
#     gd.configure_selection(
#         selection_mode="multiple",
#         use_checkbox=True,
#         pre_selected_rows=pre_selected_idx,
#     )
#     gridoptions = gd.build()

#     grid_table = AgGrid(
#         display_df,
#         gridOptions=gridoptions,
#         columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
#         enable_enterprise_modules=False,
#     )
#     selected_idx = [r['_selectedRowNodeInfo']['nodeRowIndex'] for r in grid_table["selected_rows"]]
#     labels = np.zeros(len(df), dtype=int)
#     labels[selected_idx] = 1
#     df["label"] = labels
#     return df


# def build_annotator_radio(df, text_row="inst", show_claim=True):
#     """
#     Annotate a dataframe with st.radio. Need to load `annotator.css` for improved
#     display
#     """
#     row_buttons = []
#     failure_opts = {0: "✔️", 1: "❌", 2: "❓"}
#     with st.container():
#         st.write(
#             f"Label examples. Use {failure_opts[0]} for a **passing** test, "
#             f"{failure_opts[1]} for a **related failing test**, and"
#             f"{failure_opts[2]} for an **unrelated failing test**."
#         )
#         for idx, row in df.iterrows():
#             if show_claim:
#                 label = f"*{row.claim}* → {row[text_row]}"
#             else:
#                 label = row[text_row]
#             row_buttons.append(
#                 st.radio(
#                     label=label,
#                     index=int(getattr(row, "pred", 0)),
#                     options=failure_opts,
#                     format_func=lambda x: failure_opts[x],
#                     horizontal=True,
#                 )
#             )
#     df["label"] = row_buttons
#     return df


################
# Loaders
################
# @st.cache_resource
def load_spacy(spacy_model="en_core_web_sm"):
    return spacy.load(spacy_model)


# @st.cache_resource
def load_fasttext():
    if not Path(f"cc.en.300.bin").exists():
        # st.write("One-time download of fasttext model...")
        fasttext.util.download_model("en", if_exists="ignore")
    return fasttext.load_model("./cc.en.300.bin")


session_state = {}


def load_classifier(
    classifier_type,
    classifier_encoder_model,
    use_differentiable_head=False,
    num_iterations=20,
    body_learning_rate=1e-5,
    batch_size=20,
):
    """
    Initialize classifier or load it from the session state.
    We do not use `@st.cache_resource` because model parameters are updated
    and therefore not thread-safe
    """
    if "claim_classifier" not in session_state:
        if classifier_type == "cross_encoder":
            session_state["claim_classifier"] = ClaimClassifierCE(
                model_type=classifier_encoder_model,
                predict_without_fit=True,
            )
        elif classifier_type == "setfit":
            session_state["claim_classifier"] = ClaimClassifierSF(
                model_type=classifier_encoder_model,
                use_differentiable_head=use_differentiable_head,
                num_iterations=num_iterations,
                body_learning_rate=body_learning_rate,
                batch_size=batch_size,
                predict_without_fit=True,
            )
        else:
            raise NotImplementedError("Classifier type must be either `setfit` or `cross_encoder`")
    return session_state["claim_classifier"]


# @st.cache_resource
def load_tokenizer(model_type):
    return AutoTokenizer.from_pretrained(model_type)


class EOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_tokens: List[str], tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.eos_tokens = set(eos_tokens)
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # have to convert at point of generation because of tokens like `."`
        last_token = self.tokenizer.convert_ids_to_tokens(input_ids[:, -1].item())
        return any(t in last_token for t in self.eos_tokens)


# @st.cache_resource(show_spinner="Loading target generator...")
def load_hf_generator(model_type):
    tokenizer = load_tokenizer(model_type)
    eos_stopping_criteria = EOSStoppingCriteria(["\n", "."], tokenizer)

    generator = pipeline(
        "text-generation",
        model=model_type,
        top_p=0.95,
        do_sample=True,
        max_new_tokens=50,
        stopping_criteria=[eos_stopping_criteria],
        device=0 if torch.cuda.is_available() else torch.device("cpu"),
    )

    if generator.model.config.pad_token_id is None and generator.model.config.eos_token_id is not None:
        generator.model.config.pad_token_id = generator.model.config.eos_token_id
        generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id

    return generator


##################
# Data processing
##################
# @st.cache_data
def _store_annotations(gen_df, target_model, is_test=True):
    """
    Store the labels in the streamlit session state, `is_test` is an indicator
    to separate test annotations (i.e., of a target model) from others.
    """
    assert {"claim", "inst", "label"} == set(gen_df.columns)
    gen_df = gen_df.dropna(subset=["label"]).assign(  # drop any unlabeled data
        is_test=is_test, target_model=target_model
    )
    # if "annotated" not in st.session_state:
    #     st.session_state["annotated"] = gen_df
    # else:
    #     # add annotations to a persistent state
    #     st.session_state["annotated"] = (
    #         pd.concat([st.session_state["annotated"], gen_df])
    #           .drop_duplicates(subset=["claim", "inst"], keep="last")
    #     )


def store_annotations(gen_df, target_model, is_test=True):
    gen_df.sort_index(inplace=True)  # sort to correctly trigger cache
    _store_annotations(gen_df[["claim", "inst", "label"]], target_model, is_test)


def build_completion_df(test_data, completions, target_model):
    """Create the dataframe, then pull existing annotations"""
    completion_df = pd.DataFrame(
        [
            # claim, inst (for training), completion (for review), target (for reference)
            (claim, f"{prompt}{comp}", f"{prompt}[{comp}]", target)
            for (claim, (prompt, target)), completions in zip(test_data, completions)
            for comp in completions  # can be multiple
        ],
        columns=["claim", "inst", "completion", "target"],
    )
    completion_df = completion_df.drop_duplicates(subset=["claim", "inst"])
    # if "annotated" in st.session_state:
    #     # TODO: merging is probably slow---see if possible to speed up (poss w/ index)
    #     annotated_df = st.session_state["annotated"]
    #     completion_df["is_test"] = True
    #     completion_df["target_model"] = target_model
    #     completion_df = completion_df.merge(
    #         annotated_df, how="outer", on=["claim", "inst", "is_test", "target_model"]
    #     )

    return completion_df


#####################
# Claim creation
#####################
# Generating inferences
PREDICATES = [
    # infinitive, 2nd pp, 3rd pp
    ("be", "is", "are"),
    ("do", "does", "do"),
    ("want", "wants", "want"),
    ("cause", "causes", "cause"),
]

# can add P -> M "{p_inf}" and negP ->  M "not" "{p_inf}" if desired
CLAIM_GRAMMARS = {
    # be; "cancer is contagious", "people are dumb"
    (
        "be",
    ): """
        CLAIM -> S P O | D S P O | "some" S negP O
        D -> "not all" | "most"
        M -> "should" | "must"
        S -> "{s}"
        P -> "{p}"
        negP -> "{p}" "not"
        O -> "{o}"
    """,
    # do, want, cause; "vaccines cause autism", "people want to die", "people do bad things"
    (
        "do",
        "want",
        "cause",
    ): """
        CLAIM -> S P O | D S P O | "some" S negP O
        D -> "not all" | "most"
        M -> "should" | "must"
        Q -> "always" | "only"
        S -> "{s}"
        P -> "{p}" | Q "{p_inf}"
        negP -> "do not" "{p_inf}"
        O -> "{o}"
    """,
}


def _parse_claim(claim, allowed_predicates):
    """Return S, P, O triple from a claim in correct format"""
    if isinstance(claim, str):
        claim = claim.lower()
        pred_regex = re.compile(" " + " | ".join(allowed_predicates) + " ")
        found_pred = pred_regex.findall(claim)
        if len(found_pred) != 1:
            raise ValueError("Could not find a valid verb in the claim")
        s, p, o = [i.strip() for i in re.split(f"({found_pred[0]})", claim)]
    else:
        s, p, o = [i.lower().strip() for i in claim]
    return s, p, o


def generate_inferences(claim, predicates, grammars):
    """
    Given a claim in subject, predicate, object form, generate additional claims
    """
    # unpack grammars and predicates
    grammars = {i: g for infs, g in grammars.items() for i in infs}
    predicates = {c: conjs[0] for conjs in predicates for c in conjs[1:]}

    # extract the <S,P,O> triple
    try:
        s, p, o = _parse_claim(claim, predicates)
    except ValueError:
        return []

    p_inf = predicates[p]
    filled_grammar_template = grammars[p_inf].format(s=s, p=p, o=o, p_inf=p_inf)
    prop_grammar = nltk.CFG.fromstring(filled_grammar_template)
    return [" ".join(s) for s in generate(prop_grammar)]


# Based on similar words
# @st.cache_data
def find_related_claims(claim, predicates, _ft_model=None, subj=True, obj=True, k=10):

    # extract the triple
    predicates = {c: i == 1 for conjs in predicates for i, c in enumerate(conjs[1:])}
    try:
        s, p, o = _parse_claim(claim, predicates)
    except ValueError:
        return []
    is_plural = predicates[p]

    # get related terms
    related_s, related_o = [s], [o]
    if subj:
        related = find_related_words(s, ft_model=_ft_model, ft_k=k, wn_k=k)
        related = list(set(related["nearest_vecs"] + related.get("hyponyms", [])))
        if is_plural:
            eng = inflect.engine()
            related = [eng.plural_noun(w) if not eng.singular_noun(w) else w for w in related]
        related_s += related
    if obj:
        related = find_related_words(o, ft_model=_ft_model, ft_k=k, wn_k=k)
        related_o += list(set(related["nearest_vecs"] + related.get("hyponyms", [])))

    rel_props = [f"{rel_s} {p} {rel_o}" for rel_s, rel_o in itertools.product(related_s, related_o)]
    return rel_props


def find_related_words(word, ft_model=None, ft_k=10, wn_k=20):
    data = {}
    # populate nearest vectors
    nns = []
    k_attempt = ft_k
    punct = re.compile(r"[\.,!\-]")
    while len(nns) < ft_k and k_attempt < 10_000:
        nns = [
            w
            for _, w in ft_model.get_nearest_neighbors(word, k=k_attempt)
            if word not in w.lower() and not punct.search(w)  # avoid returning the same word
        ][:ft_k]
        k_attempt = k_attempt * 5
    data["nearest_vecs"] = nns

    # populate hyponyms
    synsets = wordnet.synsets(word)
    if synsets:
        hyponyms = [lem.name() for hyp in synsets[0].hyponyms() for lem in hyp.lemmas()]
        hyponyms = list(set(hyponyms))  # de-dup
        word_vec = ft_model[word]
        dists = [cosine(word_vec, ft_model[w]) for w in hyponyms]
        top_idx = np.argsort(dists)[:wn_k]
        data["hyponyms"] = [hyponyms[i] for i in top_idx]  # TODO: explore other wordnet attrs
        data["_synset_def"] = synsets[0].definition()

    # TODO: ensure same POS
    return data


# @st.cache_data
def correct_grammar_with_gpt3(phrase, seed=11235, engine="gpt-3.5-turbo-instruct"):
    """Render a phrase grammatically correct w/ GPT-3"""
    grammar_cmd = "Change the language of the statement to standard English:"
    grammar_exs = {
        "Injury are bad": "Injuries are bad",
        "Horse_hides are soft": "Horse hides are soft",
        "i love fruit": "I love fruit",
        "you are buaetiful": "You are beautiful",
        "Cows are minerals": "Cows are minerals",
    }
    prompt = prompt_openai.make_prompt(phrase, grammar_cmd, grammar_exs, seed=seed, engine=engine)
    return prompt_openai.complete_prompt(prompt, n=1, temperature=0)[0][0]


#####################
# Test creation
#####################
# @st.cache_data(show_spinner="Creating tests with GPT-3...")
def truncate_text_by_length(texts, _tokenizer=None, n=0.5, by_tokens=True):
    """
    Truncate input text to create a prompt. If `n` is an integer, used
    as an index slice. Otherwise, treated as a porportion of tokens

    If `by_tokens` is True, then split on tokens (per `tokenizer`), else
    split on whitespace.
    """
    prompts_and_targets = []
    for text in texts:
        if by_tokens:
            toks = _tokenizer.encode(text)
            idx = int(len(toks) * n) if np.abs(n) < 1 else n
            p_and_t = _tokenizer.decode(toks[:idx]), _tokenizer.decode(toks[idx:])
            prompts_and_targets.append(p_and_t)
        else:
            words = text.split(" ")
            idx = int(len(words) * n) if np.abs(n) < 1 else n
            p_and_t = " ".join(words[:idx]), " " + " ".join(words[idx:])
            prompts_and_targets.append(p_and_t)
    return prompts_and_targets


# @st.cache_data
def truncate_text_by_root(texts, _spacy_model="en_core_web_sm", use_last_root=True):
    """
    Truncate input text based on the root verb to create a prompt.

    If `use_last_root`, then the last root verb found is used.
    Otherwise, the first one is used.
    """
    if isinstance(_spacy_model, str):
        _spacy_model = spacy.load(_spacy_model)

    prompts_and_targets = []
    use_last_root = -1 * use_last_root  # -1 or 0
    for doc in _spacy_model.pipe(texts):
        root = [tok for tok in doc if tok.dep_ == "ROOT"][use_last_root]
        # check if the root is negated after its position in the sentence
        neg_is = [c.i for c in root.children if c.dep_ == "neg"]
        root_i = max([root.i] + neg_is)
        p_and_t = doc[: root_i + 1].text, " " + doc[root_i + 1 :].text
        prompts_and_targets.append(p_and_t)
    return prompts_and_targets


# @st.cache_data(show_spinner="Truncating with GPT-3...")
def truncate_text_with_gpt3(texts, claims, instruction=None, few_shot_exemplars=None, engine="text-danvinci-002"):
    """
    Truncate input text to create a prompt using GPT-3
    """
    # TODO: this could use work
    if instruction is None or few_shot_exemplars is None:
        data = exemplars.FEW_SHOT_INTERNAL["generations_to_tests"]
        instruction = instruction or data["instruction"]
        few_shot_exemplars = few_shot_exemplars or data["exemplars"]

    gpt_prompts = [
        prompt_openai.make_prompt(f"Claim: {claim} | Sentence: {text}", instruction, few_shot_exemplars)
        for text, claim in zip(texts, claims)
    ]
    batch_size = prompt_openai._OPENAI_MAX_PROMPTS
    prompts_and_targets = []
    for i in range(0, len(gpt_prompts), batch_size):
        prompt_batch = gpt_prompts[i : i + batch_size]
        text_batch = texts[i : i + batch_size]

        test_prompts = prompt_openai.complete_prompt(
            prompt_batch, n=1, temperature=0.5, engine=engine, return_logp=False
        )
        prompts_and_targets.extend(
            (prompt, text.replace(prompt, "")) for (prompt, _), text in zip(test_prompts, text_batch) if prompt in text
        )
    return prompts_and_targets


# @st.cache_data(show_spinner="Generating tests...")
def generate_from_prompts_hf(prompts, _generator, repeats=1):
    """
    Generate text from `_generator` using the prompts
    """
    # TODO: add attention mask?
    outputs = [
        [o[0]["generated_text"].replace("\n", " ") for o in _generator(prompts, return_full_text=False)]
        for _ in range(repeats)
    ]
    return list(zip(*outputs))  # TODO: correct format?


def generate_from_prompts_gpt3(prompts, engine, repeats=1, use_pb=True):
    """
    Generate text from a gpt-3 model given the promtps
    """
    # TODO: move to `prompt_openai.py`, clean up
    batch_size = prompt_openai._OPENAI_MAX_PROMPTS
    completions = []
    # if use_pb:
    #     pb = st.progress(0.)
    for idx in range(0, len(prompts), batch_size):
        test_batch = prompts[idx : idx + batch_size]
        completions.extend(
            prompt_openai.complete_prompt(test_batch, n=repeats, top_p=0.95, max_tokens=50, engine=engine)
        )
        # if use_pb:
        #     pb.progress((min(len(prompts), idx+batch_size))/len(prompts))
    return [
        tuple(c for c, logp in completions[j : j + repeats]) for j in range(0, len(completions), repeats)
    ]  # must collaspe back down to account for repeats
