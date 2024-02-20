import json
import math
import time
import numpy as np
import numpy.random as npr
from contextlib import nullcontext
from functools import partial
from rich.live import Live
from copy import deepcopy

from langchain.llms import BaseLLM

from params import AllParams
from constants import Dataset as D
from tools.utils import Logger
from tools.track import get_progress_bar
from constants import mwp_datasets, LLM, Dataset as D, chat_lms
from prompts.base import ExampleTemplate
from prompts.few_shot import FewShotPromptTemplate2

def complete_prompts(params, llm, examples, prompts, sep, example_template):
    is_classification = hasattr(example_template, 'get_choices')
    if not is_classification:    # generation
        if params.lm_name not in chat_lms:
            response = llm.generate(prompts, stop=[sep])
            llm_outputs = [gen[0].text for gen in response.generations]
        else:   # turbo
            response = llm._generate(prompts, stop=[sep])
            llm_outputs = [gen[0].text for gen in response.generations]
            # raise NotImplementedError
            # messages = [[example_template.prepare_for_turbo(e)
            #                 for e in p.split(sep)] for p in prompts]
            # response = llm._generate(messages, stop=[sep])
            # llm_outputs = [gen[0].text for gen in response.generations]
    else:   # classification
        if params.lm_name not in chat_lms:
            choices = [example_template.get_choices(**ex) for ex in examples]
            llm_outputs = llm._classify_v3(prompts=prompts, choices=choices)
        else:
            response = llm.generate(prompts, stop=[sep])
            llm_outputs = [gen[0].text for gen in response.generations]
            # raise NotImplementedError

    return llm_outputs

def evaluate_prompt(params, ex, res, prompt, demos, example_template, tokenizer):
    prompt_metrics = dict(n_shots=len(demos))
    # if params.dataset not in no_prompt_cov_datasets:
    #     prompt_metrics |= prompt_coverage_fn(ex=ex, demos=demos)
    is_classification = hasattr(example_template, 'get_choices')
    if is_classification:
        res['_target'] = example_template.get_target(example_template.get_choices(**ex), **ex)
        demo_targets = [
            example_template.get_target(example_template.get_choices(**d), **d)
            for d in demos]
        res['demo_targets'] = demo_targets
        npr.shuffle(demo_targets)
        res['majority_pred'] = max(set(demo_targets), key=demo_targets.count)
        majority_eval_metrics = example_template.check_output(res['majority_pred'], **ex)
        prompt_metrics |= {f'majority_{k}': v for k, v in majority_eval_metrics.items()}
        prompt_metrics['majority_precision'] = 100 * np.mean(
            [t == res['_target'] for t in demo_targets])
    orig_prompt = prompt
    if tokenizer:
        res['orig_prompt'] = orig_prompt
        prompt = tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
    res['prompt'] = prompt
    return prompt_metrics

def evaluate_completion(params, ex, res, llm_output, example_template, tokenizer):
    is_classification = hasattr(example_template, 'get_choices')
    res['completion'] = llm_output
    # Compute Evaluation Metrics
    if not is_classification:
        pred = example_template.parse_output(llm_output.strip(), **ex)
        target = example_template.get_target(**ex)
        if tokenizer and not params.dataset in mwp_datasets:
            target = tokenizer.decode(tokenizer.encode(target), skip_special_tokens=True)
        res |= dict(pred=pred, _target=target)
        eval_metrics = example_template.check_output(res['pred'], target, **ex)
    else:
        target = example_template.get_target(example_template.get_choices(**ex), **ex)
        res |= dict(pred=llm_output, _target=target)
        eval_metrics = example_template.check_output(res['pred'], **ex)
    return eval_metrics

class MetricsAggregator:
    def __init__(self):
        self.agg_metrics = {}
        self.n_total = 0

    def increment(self, ex_metrics):
        for k, v in ex_metrics.items():
            self.agg_metrics[k] = self.agg_metrics.get(k, 0) + v
        self.n_total += 1

    @property
    def normalized(self):
        return {k: v / self.n_total for k, v in self.agg_metrics.items()} | dict(n_total=self.n_total)

def eval(
    params: AllParams, test_ds, llm: BaseLLM, prompt_template: FewShotPromptTemplate2,
    sel_time: float, logger: Logger, progress=None, debug=False
):
    """Do ICL and compute prompt and performance metrics"""
    log = logger.log
    tokenizer = llm.tokenizer if hasattr(llm, 'tokenizer') else None
    example_template = prompt_template.example_template
    sep = prompt_template.example_separator

    results = []
    agg_metrics = MetricsAggregator()
    n_test, batch_size = len(test_ds), params.exp.batch_size
    n_batch = math.ceil(n_test / batch_size)
    progress = progress or get_progress_bar(console=logger.std_console)
    beg = time.time()
    with Live(progress, refresh_per_second=1, console=logger.std_console) if not debug else nullcontext():
        for batch_i in progress.track(range(n_batch), description='Evaluating..') if not debug else range(n_batch):
            log(f"Batch {batch_i+1}/{n_batch}")
            test_batch = test_ds.select(np.arange(
                batch_i * batch_size, min(n_test, (batch_i + 1) * batch_size)))

            # Get few-shot prompts
            is_turbo = params.lm_name in chat_lms
            prompts, demos_l = zip(*[
                prompt_template.format(**ex,  is_turbo=is_turbo, return_demos=True)
                for ex in test_batch])

            # Complete prompts
            llm_outputs = complete_prompts(params, llm, test_batch, prompts, sep, example_template)

            # Evaluate prompts and completions
            for ex, prompt, demos, llm_output in zip(test_batch, prompts, demos_l, llm_outputs):
                res = deepcopy(ex)
                prompt_metrics = evaluate_prompt(
                    params, ex, res, prompt, demos, example_template, tokenizer)
                eval_metrics = evaluate_completion(
                    params, ex, res, llm_output, example_template, tokenizer)

                # Aggregate Resutls
                res['metrics'] = prompt_metrics | eval_metrics
                results.append(res)
                agg_metrics.increment(res['metrics'])

                if debug:
                    comp_color = 'blue' if 'accuracy' in res['metrics'] and res['metrics']['accuracy'] else 'red'
                    log('Prompt and Completion:')
                    log(f"[green]{res['prompt']}[/green][{comp_color}]{res['completion']}[/{comp_color}]")
                    log(f'Inputs: {ex}')
            log(str(agg_metrics.normalized))
    log(ex)
    log(prompt)
    icl_time = time.time() - beg
    time_metrics = dict(sel_time=sel_time / n_test, icl_time=icl_time / n_test)
    metrics = agg_metrics.normalized | time_metrics
    log(str(metrics))
    data = dict(results=results, metrics=metrics)
    print(f"Saving results to {params.resultsfile} ..")
    json.dump(data, open(params.resultsfile, 'w'), indent=2, separators=(',', ': '))
    return data

def dump_prompts(
    params: AllParams, test_ds, prompt_template: FewShotPromptTemplate2,
    sel_time: float, logger: Logger, tokenizer = None, progress=None, debug=False
):
    log = logger.log
    results = []
    agg_metrics = MetricsAggregator()
    progress = progress or get_progress_bar(console=logger.std_console)
    with Live(progress, refresh_per_second=1, console=logger.std_console) if not debug else nullcontext():
        for ex in progress.track(test_ds, description='Creating Prompts..') if not debug else test_ds:
            res = deepcopy(ex)
            prompt, demos = prompt_template.format(**ex, return_demos=True)
            prompt_metrics = evaluate_prompt(
                params, ex, res, prompt, demos, prompt_template.example_template, tokenizer)

            if debug:
                log('Prompt and Completion:')
                log(f"[green]{res['prompt']}[/green]")
                log(f'Inputs: {ex}')

            # Aggregate Resutls
            res['metrics'] = prompt_metrics
            agg_metrics.increment(res['metrics'])
            results.append(res)
    log(ex)
    log(prompt)
    metrics = agg_metrics.normalized | dict(sel_time=sel_time / len(test_ds))
    log(str(metrics))
    data = dict(results=results, metrics=metrics)
    print(f"Saving results to {params.promptsfile} ..")
    json.dump(data, open(params.promptsfile, 'w'), indent=2, separators=(',', ': '))
    return data

def eval_prompts(
    params: AllParams, llm: BaseLLM, example_template, sep,
    logger: Logger, progress=None, debug=False
):
    log = logger.log
    tokenizer = llm.tokenizer if hasattr(llm, 'tokenizer') else None
    prompts_data = json.load(open(params.promptsfile))
    results = prompts_data['results']
    all_prompts = np.array([d['orig_prompt' if 'orig_prompt' in d else 'prompt'] for d in results])
    agg_metrics = MetricsAggregator()
    n_test, batch_size = len(results), params.exp.batch_size
    n_batch = math.ceil(n_test / batch_size)
    beg = time.time()
    progress = progress or get_progress_bar(console=logger.std_console)
    with Live(progress, refresh_per_second=1, console=logger.std_console) if not debug else nullcontext():
        for batch_i in progress.track(range(n_batch), description='Evaluating..') if not debug else range(n_batch):
            log(f"Batch {batch_i+1}/{n_batch}")
            idxs = np.arange(batch_i * batch_size, min(n_test, (batch_i + 1) * batch_size))
            prompts = all_prompts[idxs]
            test_batch = [results[i] for i in idxs]
            llm_outputs = complete_prompts(params, llm, test_batch, prompts, sep, example_template)
            # Evaluate prompts and completions
            for i, prompt, llm_output in zip(idxs, prompts, llm_outputs):
                res = results[i]
                ex = {k: v for k, v in res.items() if k not in ['prompt', 'metrics']}
                eval_metrics = evaluate_completion(
                    params, ex, res, llm_output, example_template, tokenizer)

                if debug:
                    log('Prompt and Completion:')
                    log(f"[green]{res['prompt']}[/green][red]{res['completion']}[/red]")
                    log(f'Inputs: {ex}')

                # Aggregate Resutls
                res['metrics'] |= eval_metrics
                agg_metrics.increment(eval_metrics)
            log(str(agg_metrics.normalized))
    log(ex)
    log(prompt)
    icl_time = time.time() - beg
    metrics = agg_metrics.normalized | dict(icl_time=icl_time / n_test)
    log(str(metrics))
    data = dict(results=results, metrics=metrics)
    print(f"Saving results to {params.resultsfile} ..")
    json.dump(data, open(params.resultsfile, 'w'), indent=2, separators=(',', ': '))
    return data

def prompt_to_demos(prompt, prefix, example_template: ExampleTemplate = None):
    """Unused"""
    if prefix.format():
        demos = prompt.split('\n\n')[1:-1]
    else:
        demos = prompt.split('\n\n')[:-1]
    def undo_example_template(demo_str):
        source_str, target_str = demo_str.split('\n')
        source = source_str[source_str.find(': ') + len(': '):]
        target = target_str[target_str.find(': ') + len(': '):]
        return dict(source=source, target=target)
    undo_fn = example_template.undo_format if example_template else undo_example_template
    return [undo_fn(d) for d in demos]

def prompt_coverage(ex, substruct_fns, prefix, example_template, prompt=None, demos=None, n_shots:int = None):
    """Unused"""
    from selector.base import bag_relevance
    coverage = {}
    if demos is None:
        demos = prompt_to_demos(prompt, prefix, example_template)
        demo_sources = [d['source'] for d in demos]
        demo_targets = [d['target'] for d in demos]
    else:
        demo_sources = [example_template.get_source(**d) for d in demos]
        demo_targets = [example_template.get_target(**d) for d in demos]
    test_source = example_template.get_source(**ex)
    test_target = example_template.get_target(**ex)
    assert n_shots is None or len(demos) == n_shots
    # assert test_source == prompt.split('\n\n')[-1].split('\n')[0][len('Sentence: '):], prompt
    for substruct, substruct_fn in substruct_fns.items():
        if 'lf' not in 'substruct':
            test_bag = substruct_fn([test_source])[0]
            demos_bag = set([s for bag in substruct_fn(demo_sources) for s in bag])
        else:
            test_bag = substruct_fn([test_target])[0]
            demos_bag = set([s for bag in substruct_fn(demo_targets) for s in bag])
        coverage[f'{substruct}_recall'] = 100 * bag_relevance(test_bag, demos_bag, 'recall')
    return coverage

def get_substruct_fns(lfst:bool = True):
    """Unused"""
    from tools.structure.substructs import get_parser, get_substructs
    # from selector.base import SelectorUtilsMixin, StructuralSelectorArgs as Args, get_parser, bag_relevance
    # get_substructs = SelectorUtilsMixin.get_substructs
    get_args = lambda substruct, size: dict(substruct=substruct, subst_size=size)
    substruct_fns = {
        'ngram_1': partial(get_substructs, **get_args('ngram', 1)),
        'ngram_4': partial(get_substructs, **get_args('ngram', 4)),
        'depst_4': partial(get_substructs, **get_args('depst', 4), parser=get_parser('spacy')),
    }
    if lfst:
        substruct_fns['lfst_4'] = partial(get_substructs, **get_args('lfst', 4))
    return substruct_fns

def lf_unigram_coverage(res, metric='f1'):
    from tools.structure.ast_parser import tokenize_lf
    pred_bag = set(tokenize_lf(res['pred']))
    target_bag = set(tokenize_lf(res['target']))
    common = pred_bag & target_bag
    recall = len(common) / len(target_bag)
    if metric == 'recall': return recall
    precision = len(common) / len(pred_bag)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def ftk(res):
    from tools.structure.ast_parser import target_to_ast
    from tools.structure.ftk import normalized_ftk
    try:
        pred_ast = target_to_ast(res['pred'])
    except:
        return 0
    target_ast = target_to_ast(res['target'])
    return normalized_ftk(target_ast, pred_ast)

def em(res):
    return res['pred'] == res['target']
