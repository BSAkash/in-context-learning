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

    if SP.selector_type == ES.RANDOM:
        fewshot_prompt = fewshot_prompt_fn(examples=list(train_ds.select(np.arange(SP.n_shots))))
        sel_time = 0
    else:
        ex_len_fn = lambda ex, **kwargs: enc_len_fn(templates.example_template.format(**ex, **kwargs))
        ex_template = templates.selection_example_template
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
