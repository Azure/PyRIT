import random
from typing import Optional, Union, Dict

import numpy as np
from openai import OpenAI

client = OpenAI()

from pyrit.prompt_converter.utils import st_cache_data
# from utils import st_cache_data


_OPENAI_MAX_PROMPTS = 20
_OPENAI_MAX_SAMPLES = 10


def make_prompt(
    instance: str, 
    instruction: str,
    few_shot_exemplars: Optional[Dict[str, Union[list, str]]] = None,
    one_output_per_exemplar: bool = False,
    sample_exemplars: Optional[int] = None,
    sample_suffixes: Optional[int] = None,
    seed=None,
):
    """
    Make a randomized prompt from instructions and few shot examplars

    `instance`: the example we are doing inference for
    `instruction`: a natural language instruction that appears before the examplars
    `few_shot_exemplars`: a dictionary of input-output examplars
    `one_output_per_exemplar`: if multiple outputs are provided per input as a list, then
        inputs will be repeated for each output, else, concatenated with "|"
    `subsample_exemplars`: number of few-show exemplars to sample
    `sample_suffixes`: number of outputs to sample  
    """
    prompt = ''
    random.seed(seed)
    
    if instruction:
        prompt += f"{instruction}\n-------\n"
    if few_shot_exemplars is not None:
        if isinstance(few_shot_exemplars, dict):
            few_shot_exemplars = list(few_shot_exemplars.items())

        random.shuffle(few_shot_exemplars)
        n = sample_exemplars or 1e10
        exemplar_strings = []
        for input, outputs in few_shot_exemplars:
            # to avoid repetitive output, do not use exemplars that are the same as the instance
            if input == instance:
                continue

            # sample within the possible outputs, if multiple
            if isinstance(outputs, (list, tuple)):
                if sample_suffixes is None:
                    k = len(outputs)
                else:
                    k = min(sample_suffixes, len(outputs))

                outputs = random.sample(outputs, k)
                if not one_output_per_exemplar:
                    outputs = [" | ".join(outputs)]
            else:
                outputs = [outputs]

            # clean out any newlines which will break things
            exemplar_strings.extend(f"{input}->{output}".replace("\n", " ") for output in outputs)
            if len(exemplar_strings) >= n:
                break
        prompt += "\n".join(exemplar_strings) + "\n"
    if instance:
        prompt += instance + ("->" * ("->" not in instance))
    return prompt


def filter_results(results, split_output=True):
    """
    Remove duplicates and filter probable results a GPT-3 prompt
    """
    if split_output:
        results = [(sent.strip(), logp) for out, logp in results for sent in out.split("|")]
    r = sorted(results, key=lambda x: -x[1])
    ret = []
    in_ret = set()
    for x in r:
        text = x[0].rstrip(".").strip().lower()
        if text in in_ret or not text:
            continue
        in_ret.add(text)
        ret.append(x)
    return ret


@st_cache_data(show_spinner=False)
def complete_prompt(prompts, n=5, top_p=1, temperature=1,  max_tokens=200, stop='\n', engine="gpt-3.5-turbo-instruct", return_logp=True):
    """Pass one or more prompts to GPT-3 and calculate their log-probability"""
    if temperature < 1 and top_p < 1:
        raise ValueError("Set either temperature or top_p, but not both")

    response = client.completions.create(model=engine,
    prompt=prompts,
    max_tokens=max_tokens,
    n=n,
    stop=stop,
    logprobs=1,
    top_p=top_p,
    temperature=temperature)
    # calculate log-prob (accounting for log-probs returned  after `stop`)
    lines, scores = [], []
    for choice in response.choices:
        text = choice.text
        log_prob = 0
        if return_logp:
            to_compare = ""
            for tok, tok_prob in zip(choice.logprobs.tokens, choice.logprobs.token_logprobs):
                if to_compare == text:
                    break
                to_compare += tok
                log_prob += tok_prob
        lines.append(text)
        scores.append(log_prob)

    n_prompts = 1 if isinstance(prompts, str) else len(prompts)
    assert(len(lines) == n*n_prompts)

    return list(zip(lines, scores))


@st_cache_data
def run_pipeline(
    instance,
    instruction,
    few_shot_exemplars,
    target_n,
    one_output_per_exemplar=False,
    top_p=0.95,
    temperature=1,
    stop="\n",
    engine="gpt-3.5-turbo-instruct"
):
    # determine how many prompts to create
    if one_output_per_exemplar:
        mean_outputs_per_exemplar = 1
    else:
        mean_outputs_per_exemplar = np.mean([
            len(ex) if isinstance(ex, (tuple, list)) else 1 
            for ex in few_shot_exemplars.values()
        ])
    n_low, n_high = target_n_to_prompt_n(target_n, 1, mean_outputs_per_exemplar)

    # make the prompts and query the model
    prompts  = [
        make_prompt(instance, instruction, few_shot_exemplars, one_output_per_exemplar=one_output_per_exemplar)
        for _ in range(n_high)
    ]
    a = complete_prompt(prompts, n=n_low, top_p=top_p, temperature=temperature, stop=stop, engine=engine)
    return filter_results(a, split_output=not one_output_per_exemplar)


@st_cache_data(show_spinner=False)
def run_pipeline_per_source(
    instance, # target instance
    few_shot_sources,
    default_instruction=None,
    target_n=None,
    exemplars_per_prompt=8, # maximum exemplars in a prompt
    outputs_per_exemplar=4, # maximum outputs mapped to a given input
    one_output_per_exemplar=False,
    top_p=0.95,
    temperature=1,
    stop="\n",
    engine="gpt-3.5-turbo-instruct",
    batch_size=_OPENAI_MAX_PROMPTS,
):
    """
    Query GPT-3 for a given `instance` using a collection of `few_shot_sources`
    """
    if one_output_per_exemplar:
        mean_outputs_per_exemplar = 1
    else:
        mean_outputs_per_exemplar = np.mean([
            min(outputs_per_exemplar, len(ex)) if isinstance(ex, (tuple, list)) else 1 
            for few_shot_data in few_shot_sources.values()
            for ex in few_shot_data["exemplars"].values()
        ])
    n_low, n_high = target_n_to_prompt_n(
        target_n, len(few_shot_sources), mean_outputs_per_exemplar
    )
    prompts = [
        make_prompt(
            instance,
            few_shot_data.get("instruction", default_instruction),
            few_shot_data["exemplars"],
            one_output_per_exemplar=one_output_per_exemplar,
            sample_exemplars=exemplars_per_prompt,
            sample_suffixes=outputs_per_exemplar,
        )
        for few_shot_data in few_shot_sources.values()
        for _ in range(n_high)
    ]
    results = []
    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i:i+batch_size]
        r = complete_prompt(
            prompt_batch,
            n=n_low,
            top_p=top_p,
            temperature=temperature,
            stop=stop,
            engine=engine
        )
        results.extend(r)
    return filter_results(results, split_output=not one_output_per_exemplar)


def target_n_to_prompt_n(
    target_n,
    n_few_shot_sources=1,
    mean_outputs_per_exemplar=1,
    expected_duplicate_prob=0.8,
):
    """
    Helper function to infer the number of prompts you need to achieve the desired
    number of generated outputs
    """
    target_n = (
        target_n
        / n_few_shot_sources
        / mean_outputs_per_exemplar
        * (1 / expected_duplicate_prob)
    )
    sqrt_n = np.sqrt(target_n)
    n_samples, n_prompts = max(1, int(np.floor(sqrt_n))), int(np.ceil(sqrt_n))
    if n_samples > _OPENAI_MAX_SAMPLES:
        n_samples = _OPENAI_MAX_SAMPLES
        n_prompts = target_n // n_prompts
    return n_samples, n_prompts