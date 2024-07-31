import os
import torch
import numpy as np
import hydra

from datasets import Dataset
from copy import deepcopy
from functools import partial
from omegaconf import OmegaConf
from rich import print


from tools.utils import Logger
from tools.lm import get_enc_len_fn
from params import AllParams
from constants import Dataset as D, ExSel as ES, LLM
from prompts.few_shot import FewShotPromptTemplate2
from eval import eval, dump_prompts
from data_params import Templates
from prompts.base import ExampleTemplate

def set_seeds(seed):
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed

SEPARATOR = '\n\n'

def get_max_target_length(
    dataset: Dataset, example_template, llm: LLM = None, enc_len_fn = None):
    enc_len_fn = enc_len_fn or get_enc_len_fn(llm)
    test_strings = [example_template.format(**ex, test=True) for ex in dataset]
    completed_strings = [example_template.format(**ex, test=False) for ex in dataset]
    test_str_lens = [enc_len_fn(s) for s in test_strings]
    completed_str_lens = [enc_len_fn(s) for s in completed_strings]
    target_lens = [c - t for t, c in zip(test_str_lens, completed_str_lens)]
    return max(target_lens)

def get_selector(
    P: AllParams, candidates: Dataset, test_ds: Dataset, example_template: ExampleTemplate,
    ex_len_fn=None, max_len=-1, subtract_gen_len=False, return_time=False,
):
    """Get the selector based on the given selector parameters `P.selector`

    Args:
        P (AllParams):
        candidates: the pool of candidate examples to select from
        test_ds: the test instances so the selectors can preselect the shots faster using batching
        example_template: template to convert instances to text for use in selection
        ex_len_fn: function to compute tokenized length of examples in an ICL prompt.
        max_len: _description_. limit the number of demonstrations to select based on the available context length
    """
    from selector import BertScoreSelector, GistBertScoreSelector, CosineCoverageSelector, StructuralCoverageSelector, LFCoverageSelector
    selector_type = P.selector.selector_type
    common_args = dict(
        args=P.selector, examples=candidates, query_examples=test_ds, example_template=example_template,
        ex_len_fn=ex_len_fn, max_len=max_len, subtract_gen_len=subtract_gen_len)
    device = f"cuda:{P.gpu}" if torch.cuda.is_available() and P.gpu >= 0 else "cpu"
    if selector_type == ES.COSINE:
        ex_selector = CosineCoverageSelector.from_examples(**common_args, return_time=return_time, device=device)
    elif selector_type == ES.STRUCT:
        ex_selector = StructuralCoverageSelector.from_examples(**common_args, return_time=return_time)
    elif selector_type == ES.BERTSCORE:
        ex_selector = BertScoreSelector.from_examples(**common_args, return_time=return_time, device=device)
    elif selector_type == ES.GIST_BERTSCORE:
        ex_selector = GistBertScoreSelector.from_examples(**common_args, return_time=return_time, device=device)
    elif selector_type == ES.LF_COVERAGE:
        ex_selector = LFCoverageSelector.from_examples(**common_args)
    else:
        raise ValueError(f'Unknown selector type: {selector_type}')
    return ex_selector

def standardize_selector(selector):
    """Standardize selectors by converting lists of arrays to 2D arrays."""
    if isinstance(selector.shot_scores_l, list):
        selector.shot_scores_l = np.array(selector.shot_scores_l)
    if isinstance(selector.shot_idxs_l, list):
        selector.shot_idxs_l = np.array(selector.shot_idxs_l)
    return selector

def combine_selectors(harmful_selector, harmless_selector):
    """Combine the harmful and harmless selectors."""
    # Unpack the tuples if needed
    sel_time = 0
    if isinstance(harmful_selector, tuple):
        harmful_selector, harmful_sel_time = harmful_selector
        sel_time += harmful_sel_time
    if isinstance(harmless_selector, tuple):
        harmless_selector, harmless_sel_time = harmless_selector
        sel_time += harmless_sel_time
    
    # Ensure both selectors are of the same type
    if not isinstance(harmful_selector, type(harmless_selector)):
        raise TypeError("Selectors must be of the same type to combine")

    harmful_selector = standardize_selector(harmful_selector)
    harmless_selector = standardize_selector(harmless_selector)

    # Concatenate shot scores and indices from both selectors
    combined_shot_scores_l = np.concatenate([
        harmful_selector.shot_scores_l,
        harmless_selector.shot_scores_l
    ], axis=1)  # Ensure axis is set to concatenate along the correct dimension

    combined_shot_idxs_l = np.concatenate([
        harmful_selector.shot_idxs_l,
        harmless_selector.shot_idxs_l + len(harmful_selector.demo_candidates)
    ], axis=1)

    # Sort combined scores and indices in descending order
    sorted_indices = np.argsort(combined_shot_scores_l, axis=1)[:, ::-1]  # Descending order

    # Apply sorting to scores and indices
    sorted_scores = np.take_along_axis(combined_shot_scores_l, sorted_indices, axis=1)
    sorted_idxs = np.take_along_axis(combined_shot_idxs_l, sorted_indices, axis=1)

    # Combine datasets
    from datasets import concatenate_datasets
    combined_demo_candidates = concatenate_datasets([
        harmful_selector.demo_candidates,
        harmless_selector.demo_candidates
    ])


    # Create a combined selector object
    combined_selector = deepcopy(harmless_selector)  # Start with one of the selectors
    combined_selector.shot_scores_l = sorted_scores
    combined_selector.shot_idxs_l = sorted_idxs
    combined_selector.demo_candidates = combined_demo_candidates
    combined_selector.query2idx = {**harmful_selector.query2idx, **harmless_selector.query2idx}
    
    # print('Combined selector: ', combined_selector)
    # print(f"combined_score: {sorted_scores}")
    # print(f"combined_indexes:{sorted_idxs}")

    """ check the order logic of combined examples
    for idx in combined_selector.shot_idxs_l:
        example_dataset = combined_selector.demo_candidates.select(idx)
        for idx, data in enumerate(example_dataset):
            print(f"Example {idx +  1}:")
            print(data)
            print("-" * 40)
    """
    return combined_selector, sel_time if sel_time else combine_selectors

def get_prompt_template(
    P: AllParams, train_ds: Dataset, test_ds: Dataset, candidates: Dataset,
    templates: Templates, max_new_tokens: int, logger: Logger, return_time=False,
) -> tuple[FewShotPromptTemplate2, int]:
    """return the few-shot prompt template for constructing prompts for every test instance."""
    EP, DP, LP, SP = P.shorthand
    from constants import context_length_limit
    max_len = context_length_limit[LP.lm_name] - max_new_tokens
    subtract_gen_len = False
    enc_len_fn = get_enc_len_fn(LP.lm_name)
    fewshot_prompt_fn = partial(FewShotPromptTemplate2,
        prefix_template=templates.prefix_template,
        example_template=templates.example_template,
        example_separator=SEPARATOR,
        max_len=max_len, enc_len_fn=enc_len_fn,
        subtract_gen_len=subtract_gen_len
    )

    if SP.n_shots == -1:
        P = deepcopy(P)
        SP.n_shots = 50

    harmful_candidates = candidates.filter(lambda example: example['label'] == 'Harmful')
    harmless_candidates = candidates.filter(lambda example: example['label'] == 'Harmless')
    half_shots = SP.n_shots // 2

    if SP.selector_type == ES.RANDOM:
        if P.exp.balance:
            harmful_examples = list(harmful_candidates.select(np.arange(SP.n_shots // 2)))
            harmless_examples = list(harmless_candidates.select(np.arange(SP.n_shots // 2)))
            examples = harmful_examples + harmless_examples
            fewshot_prompt = fewshot_prompt_fn(examples=examples)
        else:
            fewshot_prompt = fewshot_prompt_fn(examples=list(train_ds.select(np.arange(SP.n_shots))))
        sel_time = 0
    else:
        ex_len_fn = lambda ex, **kwargs: enc_len_fn(templates.example_template.format(**ex, **kwargs))
        ex_template = templates.selection_example_template
        
        if P.exp.balance:
            P_half = deepcopy(P)
            P_half.selector.n_shots = half_shots
            harmful_selector = get_selector(P_half, harmful_candidates, test_ds, ex_template, ex_len_fn, max_len, subtract_gen_len, return_time=return_time)
            harmless_selector = get_selector(P_half, harmless_candidates, test_ds, ex_template, ex_len_fn, max_len, subtract_gen_len, return_time=return_time)
            # with open(f'selector_output/{P.selector.selector_type}_harmful_selector.txt', 'w') as f:
            #     f.write(str(harmful_selector))
            # with open(f'selector_output/{P.selector.selector_type}_harmless_selector.txt', 'w') as f:
            #     f.write(str(harmless_selector))
            ex_selector = combine_selectors(harmful_selector, harmless_selector)
        else:
            ex_selector = get_selector(P, candidates, test_ds, ex_template, ex_len_fn, max_len, subtract_gen_len, return_time=return_time)
        if return_time:
            ex_selector, sel_time = ex_selector
            logger.log(f'Selector took {sel_time:.2f} seconds')
        fewshot_prompt = fewshot_prompt_fn(example_selector=ex_selector)
        
    if return_time:
        return fewshot_prompt, sel_time
    else:
        return fewshot_prompt

def run_main(P: AllParams, logger: Logger):
    """Run the experiment for the given parameters `P`"""
    log = logger.log
    EP, DP, LP, SP = P.shorthand
    train_ds, candidates, test_ds = DP.get_splits(EP.data_root, 'data', tokenizer=None, seed=EP.seed)
    templates: Templates = DP.get_templates()
    DP.log_templates(test_ds[0])

    torch.cuda.empty_cache()
    max_new_tokens = get_max_target_length(test_ds, templates.example_template, LP.lm_name) + 20
    if P.promptsfile.exists() and False: # TODO: test this
        from eval import eval_prompts
        llm = P.get_lm(max_tokens=max_new_tokens)
        eval_prompts(P, llm, templates.example_template, SEPARATOR,
                     logger=logger, debug=P.exp.debug)
    else:
        prompt_template, sel_time = get_prompt_template(
            P, train_ds, test_ds, candidates, templates, max_new_tokens, logger, return_time=True)
        if P.exp.only_prompts:
            dump_prompts(P, test_ds, prompt_template, sel_time,
                logger=logger, debug=P.exp.debug)
        else:
            llm = P.get_lm(max_tokens=max_new_tokens)
            eval(P, test_ds, llm, prompt_template, sel_time,
                 logger=logger, debug=P.exp.debug)

@hydra.main(version_base=None, config_name="config")
def main(P: AllParams):
    """
    Run the experiment for the given parameters `P`.
    This can be run both programmatically and from the command-line.
    `AllParams.get_cmd` is a convenient way to get the corresponding command.
    """
    P: AllParams = OmegaConf.to_object(P)
    if P.exp.tiny:
        P.data.n_cands, P.data.n_test = 40, 20

    print(P)
    print(P.output_dir)
    os.makedirs(P.output_dir, exist_ok=True)
    logger = Logger(outfile=P.logfile if not P.exp.only_prompts else P.promptslogfile)
    try:
        run_main(P, logger)
    except Exception as e:
        import traceback
        logger.log(traceback.format_exc())
        logger.log(e)

if __name__ == '__main__':
    main()