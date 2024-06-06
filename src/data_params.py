"""
Parameters for the all the supported datasets.
Can also test the implementation using the `test` function towards the end.
The test can also be invoked from the shell as (see `test_datasets`):
```
python data_params.py --datasets "SMCALFLOW;GEOQUERY"
```

"""

from __future__ import annotations

import attr
import numpy as np
import typer
import rich
from rich.rule import Rule
from typing import Any, Callable, Type
from functools import partial
from pathlib import Path
from datasets import load_dataset, Dataset
from prompts.base import (
    ExampleTemplate,
    GenerationTemplate,
    ClassificationTemplate,
    Rouge,
)
from tools.param_impl import Parameters, DictDataClass
from constants import Task as T, Dataset as D
from tools.exp import get_datasets
from tools.lm import get_enc_len_fn

app = typer.Typer()

@attr.s(auto_attribs=True)
class Templates(DictDataClass):
    """prefix for the few-shot ICL prompt. Used for task instructions."""
    prefix_template: str = ''

    """template for individual examples in few-shot ICL prompt.
    Used in `prompts.few_shot.FewShotPromptTemplate2`."""
    example_template: ExampleTemplate = None

    """template for individual examples's input and output along with instructions."""
    instructed_example_template: ExampleTemplate = None

    """template for use in example selectors.
    typically set to instructed_example_template"""
    selection_example_template: ExampleTemplate = None

@attr.s(auto_attribs=True)
class DataParams(Parameters):
    """Parent class for all dataset's parameter classes."""
    dataset: D | None = None             # Dataset name.
    task: T | None = None               # Task category

    split: str | None = None            # Split to use for train/test.
    test_split: str | None = None       # Split to use for test.
    train_split: str | None = None      # Split to use for train.

    n_test: int = -1                 # Number of test examples to use. If -1, use full test set.
    n_cands: int | None = -1         # Number of candidates to select from. If -1, full train set.

    prefix: bool = True              # Whether to prefix the ICL prompt with task description.

    def log_templates(DP: DataParams, ex = None):
        if ex is None:
            train_ds, candidates, test_ds = DP.get_splits()
            print(candidates)
            print(test_ds)
            print(test_ds[0])
            ex = test_ds[0]
        T = DP.get_templates()
        print("ICL Example Template:")
        print(T.prefix_template)
        print(T.example_template.format(**ex))
        # print()
        # print("Instructed Example Template:")
        # print(T.instructed_example_template.format(**ex))
        print()
        print("Selection Example Template:")
        print(T.selection_example_template.format(**ex))
        print()

    def filter_by_len(self, ds: Dataset, tokenizer, max_len):
        ex_template = self.get_templates().selection_example_template
        acceptable = lambda ex: len(tokenizer.tokenize(ex_template.format(**ex, test=True))) < max_len
        ds = ds.filter(acceptable)
        return ds

    def subsample_splits(
        self, train_ds: Dataset, test_ds: Dataset,
        tokenizer=None, max_len=512, seed=0,
    ) -> tuple[Dataset, Dataset, Dataset]:
        if tokenizer and max_len:
            print(f'Before filtering by length < {max_len}: {len(train_ds)} train, {len(test_ds)} test')
            train_ds = self.filter_by_len(train_ds, tokenizer, max_len)
            test_ds = self.filter_by_len(test_ds, tokenizer, max_len)
            print(f'After filtering by length < {max_len}: {len(train_ds)} train, {len(test_ds)} test')
        n_cands, n_test = self.n_cands, self.n_test
        train_ds = train_ds.shuffle(seed=seed)
        candidates = train_ds.select(np.arange(n_cands)) if n_cands != -1 and n_cands < len(train_ds) else train_ds
        if n_test != -1 and n_test < len(test_ds):
            test_ds = test_ds.shuffle(seed=seed).select(np.arange(n_test))
        print(f'Train: {train_ds}')
        print(f'Candidates: {candidates}')
        print(f'Test: {test_ds}')
        return train_ds, candidates, test_ds

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        """return the dataset"""
        raise NotImplementedError

    def get_splits(
        self, data_root: str = '../data', dataloaders_dir: str = 'data',
        tokenizer=None, max_len=512, seed=0,
    ):
        """return the full train split, candidate pool (same as train split if n_cands==-1) and test split"""
        ds = self.get_dataset(data_root, dataloaders_dir)
        return self.subsample_splits(ds[self.train_split], ds[self.test_split], tokenizer, max_len, seed)


    def get_templates(self) -> Templates:
        raise NotImplementedError

    def get_dataset_name(self):
        """name of dataset for `AllParams.exp_path`"""
        data_name_parts = [self.dataset.name]
        if hasattr(self, 'dataset_version') and self.dataset_version != 'v0':
            data_name_parts.append(self.dataset_version)
        if hasattr(self, 'embed_context') and self.embed_context:
            data_name_parts.append('embed_context')
        return '-'.join(data_name_parts)

    def get_split_name(self):
        """name of split for `AllParams.exp_path`"""
        split_name_parts = []
        if self.split:
            split_name_parts.append(self.split)
        if self.train_split and self.train_split != 'train':
            split_name_parts.append(f'{self.train_split}-{self.test_split}')
        elif self.test_split:
            split_name_parts.append(self.test_split)
        return '-'.join(split_name_parts)
        # if self.split:
        #     return self.split
        # elif self.train_split and self.train_split != 'train':
        #     return f'{self.train_split}-{self.test_split}'
        # else:
        #     return self.test_split

    def get_prompt_name(self):
        """name of prompt for `AllParams.exp_path`"""
        prompt_name_parts = []
        # if isinstance(self, SemanticParsing):
        #     if self.prompt_version != lm_default_prompt_version:
        #         prompt_name_parts.append(self.prompt_version)
        if hasattr(self, 'icl_prompt_version') and self.icl_prompt_version != 'v0':
            prompt_name_parts.append(f'icl_prompt_{self.icl_prompt_version}')
        if hasattr(self, 'sel_prompt_version') and self.sel_prompt_version != 'v0':
            prompt_name_parts.append(f'sel_prompt_{self.sel_prompt_version}')
        if not self.prefix:
            prompt_name_parts.append('no_prefix')
        return '-'.join(prompt_name_parts)

# ---------------------------------------------------------------------------- #
#                               Semantic Parsing                               #
# ---------------------------------------------------------------------------- #

@attr.s(auto_attribs=True)
class SemanticParsing(DataParams):
    dataset: D = D.GEOQUERY
    task: T = T.SEMPARSE
    instruction: str = 'Translate the following sentences into logical forms.'
    sel_instruction: str = 'Translate this sentence into a logical form representing its meaning'
    input_feature: str = 'source'       # Name of the input feature.
    target_feature: str = 'target'      # Name of the target feature.
    output_label: str = 'Logical Form'
    sel_prompt_version: str = 'v1'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset(
            name=self.dataset.value,
            path=f'{dataloaders_dir}/semparse/semparse.py',
            data_dir=f'{data_root}/semparse')

    def get_templates(self):
        T = Templates()
        T.prefix_template = self.instruction if self.prefix else ''
        # T.example_template = GenerationTemplate(get_target=self.target_feature,
        #     templates=f'Sentence: {{{self.input_feature}}}\nLogical Form: {{_target}}')
        T.example_template = GenerationTemplate(get_target=self.target_feature,
            templates=f'{{{self.input_feature}}}\t{{_target}}')
        T.instructed_example_template = GenerationTemplate(get_target=self.target_feature,
            templates=f'{self.sel_instruction}: {{{self.input_feature}}}\n{self.output_label}: {{_target}}')
        if self.sel_prompt_version == 'v0':
            T.selection_example_template = GenerationTemplate(get_target=self.target_feature,
                templates=f'{{{self.input_feature}}}\t{{_target}}'
                # templates=dict(test=f'{{{self.input_feature}}}')
                )
        elif self.sel_prompt_version == 'v1':
            T.selection_example_template = T.instructed_example_template
        else:
            raise ValueError(f'Invalid sel_prompt_version: {self.sel_prompt_version}')
        return T

@attr.s(auto_attribs=True)
class GeoQuery(SemanticParsing):
    dataset: D = D.GEOQUERY
    split: str | None = 'iid'           # Dataset split. [iid, csl_length, csl_template_{1,2,3}, csl_tmcd_{1,2,3}]
    train_split: str = 'train'          # Huggingface dataset split for training.
    test_split: str = 'test'            # Huggingface dataset split for testing.

    def get_splits(
        self, data_root: str = '../data', dataloaders_dir: str = 'data',
        tokenizer=None, max_len=512, seed=0
    ):
        ds = self.get_dataset(data_root, dataloaders_dir)
        if self.split == 'iid':
            trn, tst = ds['standard_train'], ds['standard_test']
        else:
            trn, tst = ds[f'{self.split}_train'], ds[f'{self.split}_test']
        return self.subsample_splits(trn, tst, tokenizer, max_len, seed)

@attr.s(auto_attribs=True)
class SMCalFlowCS(SemanticParsing):
    dataset: D = D.SMCALFLOW_CS
    task: T = T.SEMPARSE
    split: str = '32_S'                  # Dataset split. [{0,8,16,32}_{S,C}]

    def get_splits(
        self, data_root: str = '../data', dataloaders_dir: str = 'data',
        tokenizer=None, max_len=512, seed=0
    ):
        from datasets import concatenate_datasets
        ds = self.get_dataset(data_root, dataloaders_dir)

        fewshot_ds = ds['fewshots'].select(np.arange(int(self.split.split('_')[0])))
        train_ds = ds['train']
        test_ds = ds['iid_test'] if 'S' in self.split else ds['comp_test']
        train_ds, candidates, test_ds = self.subsample_splits(
            train_ds, test_ds, tokenizer, max_len, seed)
        candidates = concatenate_datasets([candidates, fewshot_ds])
        return train_ds, candidates, test_ds

@attr.s(auto_attribs=True)
class Atis(SemanticParsing):
    dataset: D = D.ATIS
    task: T = T.SEMPARSE
    split: str = 'iid_0'              # Dataset split. [iid_0, template_0]

    def get_splits(
        self, data_root: str = '../data', dataloaders_dir: str = 'data',
        tokenizer=None, max_len=512, seed=0
    ):
        ds = self.get_dataset(data_root, dataloaders_dir)
        return self.subsample_splits(ds[f'{self.split}_train'], ds[f'{self.split}_test'],
                                     tokenizer, max_len, seed)

@attr.s(auto_attribs=True)
class Overnight(SemanticParsing):
    dataset: D = D.OVERNIGHT
    task: T = T.SEMPARSE
    split: str = 'socialnetwork_iid_0'  # Dataset split. [socialnetwork_iid_0, socialnetwork_template_0]
    input_feature: str = 'paraphrase'   # Name of the input feature.
    target_feature: str = 'target'      # Name of the target feature.

    def get_dataset_name(self):
        return f'{self.dataset.name}-{self.input_feature}'

    def get_splits(
        self, data_root: str = '../data', dataloaders_dir: str = 'data',
        tokenizer=None, max_len=512, seed=0
    ):
        ds = self.get_dataset(data_root, dataloaders_dir)
        return self.subsample_splits(ds[f'{self.split}_train'], ds[f'{self.split}_test'],
                                     tokenizer, max_len, seed)

@attr.s(auto_attribs=True)
class Break(DataParams):
    dataset: D = D.BREAK
    task: T = T.SEMPARSE
    train_split: str = 'train'
    test_split: str = 'validation'
    qdecomp_path: str = 'third_party/qdecomp_with_dependency_graphs'

    def __attrs_post_init__(self):
        self.n_test = self.n_test or 1000

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset(f'{dataloaders_dir}/semparse/break/break.py')

    def get_templates(self):
        from prompts.base import BreakLFEM, Accuracy
        T = Templates()
        instruction = 'Decompose the following sentences into a sequences of steps.'
        T.prefix_template = instruction if self.prefix else ''
        common_args = dict(
            get_target='decomposition',
            metrics = [Accuracy(), BreakLFEM(self.qdecomp_path)],
            result_metric='lfem',
        )
        # T.example_template = GenerationTemplate(**common_args,
        #     templates='Sentence: {question_text}\nDecomposition: {_target}')
        T.example_template = GenerationTemplate(**common_args,
            templates='{question_text}\t{_target}')
        T.instructed_example_template = GenerationTemplate(**common_args,
            templates='Decompose the sentence into a sequence of steps.: {question_text}\nDecomposition: {_target}')
        T.selection_example_template = GenerationTemplate(**common_args,
            templates='{question_text}\t{_target}'
            # templates=dict(test='{question_text}')
            )
        return T

@attr.s(auto_attribs=True)
class MTOP(SemanticParsing):
    dataset: D = D.MTOP
    task: T = T.SEMPARSE
    train_split: str = 'train'
    test_split: str = 'validation'
    input_feature: str = 'question'
    target_feature: str = 'logical_form'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('iohadrubin/mtop')


@attr.s(auto_attribs=True)
class COGS(SemanticParsing):
    dataset: D = D.COGS
    task: T = T.SEMPARSE
    train_split: str = 'train'
    test_split: str = 'dev'     # gen, dev

@attr.s(auto_attribs=True)
class CFQ(SemanticParsing):  # TODO
    dataset: D = D.CFQ
    task: T = T.SEMPARSE
    split: str = 'mcd1'
    train_split: str = 'train'
    test_split: str = 'test'
    input_feature: str = 'question'       # Name of the input feature.
    target_feature: str = 'query'      # Name of the target feature.

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('cfq', self.split, revision='4119d6a62eb78f9234a689b28ba69bfae0562bc7')

@attr.s(auto_attribs=True)
class Spider(SemanticParsing):  # TODO
    dataset: D = D.SPIDER
    task: T = T.SEMPARSE
    train_split: str = 'train'
    test_split: str = 'validation'
    input_feature: str = 'question'       # Name of the input feature.
    target_feature: str = 'query'      # Name of the target feature.
    instruction: str = 'Translate the following natural langauge questions into SQL queries.'
    sel_instruction: str = 'Translate the following natural langauge question into a SQL query'
    output_label: str = 'SQL'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('spider', revision='6232cc3fad6d54c62b3ba23a364083a98ff36a17')

# ---------------------------------------------------------------------------- #
#                                      NLI                                     #
# ---------------------------------------------------------------------------- #
@attr.s(auto_attribs=True)
class QNLI(DataParams):
    dataset: D = D.QNLI
    task: T = T.NLI
    train_split: str = 'train'
    test_split: str = 'validation'
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('glue', 'qnli', revision='fd8e86499fa5c264fcaad392a8f49ddf58bf4037')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Are the following questions answered by the sentence (Yes or No):'
        T.prefix_template = instruction if self.prefix else ''
        choices = ['Yes', 'No']
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Question: {question}\nSentence: {sentence}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='{sentence}\nCan we know "{question}" given the above sentence (Yes or No)? {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class MNLI(DataParams):
    dataset: D = D.MNLI
    task: T = T.NLI
    train_split: str = 'train'
    test_split: str = 'validation_matched'
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('glue', 'mnli', revision='fd8e86499fa5c264fcaad392a8f49ddf58bf4037')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Do the premise entail the hypotheses (Yes, Maybe, or No):'
        T.prefix_template = instruction if self.prefix else ''
        choices = ["Yes", "Maybe", "No"]
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Premise: {premise}\nHypothesis: {hypothesis}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Premise: {premise}\nDoes the above premise entail the hypothesis that "{hypothesis}" (Yes, Maybe, or No)?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class RTE(DataParams):
    dataset: D = D.RTE
    task: T = T.NLI
    train_split: str = 'train'
    test_split: str = 'validation'
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('super_glue', 'rte')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Do the premise entail the hypotheses (Yes or No):'
        T.prefix_template = instruction if self.prefix else ''
        choices = ["Yes", "No"]
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Premise: {premise}\nHypothesis: {hypothesis}?\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='{premise}\nBased on the above paragraph can we conclude that "{hypothesis}" (Yes or No)? {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class WANLI(DataParams):
    dataset: D = D.WANLI
    task: T = T.NLI
    train_split: str = 'train'
    test_split: str = 'test'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('alisawuffles/WANLI')

    def get_templates(self) -> Templates:
        T = Templates()
        # instruction = 'Are the given hypotheses "entailment", "contradiction" or "neutral" with respect to corresponding premise?'
        instruction = ''
        T.prefix_template = instruction if self.prefix else ''
        T.example_template = ClassificationTemplate(
            choices=['Yes', 'No', 'Maybe'],
            get_target=lambda _choices, **kwargs: {'entailment': 'Yes', 'contradiction': 'No', 'neutral': 'Maybe'}[kwargs['gold']],
            templates='Premise: {premise}\nHypothesis: {hypothesis}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(
            choices = ["entailment", "neutral", "contradiction"],
            get_target = lambda _choices, **kwargs: kwargs['gold'],
            templates='Premise: {premise}\nIs the hypothesis that "{hypothesis}" an entailment, contradiction or neutral with respect to the above premise?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class XNLI(DataParams):
    dataset: D = D.XNLI
    task: T = T.NLI
    train_split: str = 'train'
    test_split: str = 'validation'
    split: str = 'fr'
    language: str = 'fr'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('xnli', self.split, revision='1cdcf07be24d81f3d782038a5a0b9c8d62f76e60')

    def get_templates(self) -> Templates:
        T = Templates()
        # instruction = 'Are the given hypotheses "entailment", "contradiction" or "neutral" with respect to corresponding premise?'
        instruction = ''
        T.prefix_template = instruction if self.prefix else ''
        get_target = lambda _choices, **kwargs: _choices[kwargs['label']]
        T.example_template = ClassificationTemplate(
            choices=['Yes', 'No', 'Maybe'],
            get_target=get_target,
            templates='Premise: {premise}\nHypothesis: {hypothesis}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(
            choices = ["entailment", "neutral", "contradiction"],
            get_target=get_target,
            templates='Premise: {premise}\nIs the hypothesis that "{hypothesis}" an entailment, contradiction or neutral with respect to the above premise?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class MedNLI(DataParams):
    dataset: D = D.MEDNLI
    task: T = T.NLI
    train_split: str = 'train'
    test_split: str = 'validation'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        # return load_dataset('medarc/mednli')
        return load_dataset(
            f'{dataloaders_dir}/nli/mednli/mednli.py',
            'mednli_bigbio_te',
            data_dir=f'{data_root}/nli/mednli')

    def get_templates(self) -> Templates:
        T = Templates()
        # instruction = 'Are the given hypotheses "entailment", "contradiction" or "neutral" with respect to corresponding premise?'
        instruction = ''
        T.prefix_template = instruction if self.prefix else ''
        T.example_template = ClassificationTemplate(
            choices=['Yes', 'No', 'Maybe'],
            get_target=lambda _choices, **kwargs: {'entailment': 'Yes', 'contradiction': 'No', 'neutral': 'Maybe'}[kwargs['label']],
            templates='Premise: {premise}\nHypothesis: {hypothesis}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(
            choices=["entailment", "neutral", "contradiction"],
            get_target=lambda _choices, **kwargs: kwargs['label'],
            templates='Premise: {premise}\nIs the hypothesis that "{hypothesis}" an entailment, contradiction or neutral with respect to the above premise?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

# ---------------------------------------------------------------------------- #
#                              Sentiment Analysis                              #
# ---------------------------------------------------------------------------- #

@attr.s(auto_attribs=True)
class SST2(DataParams):
    dataset: D = D.SST2
    task: T = T.SENTIMENT
    train_split: str = 'train'
    test_split: str = 'validation'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('glue', 'sst2', revision='fd8e86499fa5c264fcaad392a8f49ddf58bf4037')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Classify the sentiment of the following reviews into Negative or Positive.'
        T.prefix_template = instruction if self.prefix else ''
        choices=["Negative", "Positive"]
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Review: {sentence}\nSentiment: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Review: {sentence}\nIs the sentiment of the above review Negative or Positive?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class SST5(DataParams):
    dataset: D = D.SST5
    task: T = T.SENTIMENT
    train_split: str = 'train'
    test_split: str = 'validation'
    icl_prompt_version: str = 'v0'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('SetFit/sst5')

    def get_templates(self) -> Templates:
        choices = ["terrible", "bad", "OK", "good", "great"]
        choices_str = ', '.join(choices)
        T = Templates()
        if self.icl_prompt_version == 'v0':
            T.prefix_template = f'Classify the sentiment of the following reviews into one of the following categories: {choices_str}.'
            T.example_template = ClassificationTemplate(choices=choices,
                templates='Review: {text}\nSentiment: {_target}')
        elif self.icl_prompt_version == 'v1':
            T.prefix_template = f'Rate the sentiment of the following review on a scale of 1 to 5, where 1 is terrible and 5 is great.'
            T.example_template = ClassificationTemplate(choices=['1', '2', '3', '4', '5'],
                templates='Review: {text}\nSentiment: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Review: {text}\nDoes the review above see the movie as terrible, bad, OK, good, or great?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class Yelp(DataParams):
    dataset: D = D.YELP
    task: T = T.SENTIMENT
    train_split: str = 'train'
    test_split: str = 'test'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('yelp_polarity')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Classify the sentiment of the following reviews into Negative or Positive.'
        T.prefix_template = instruction if self.prefix else ''
        choices=["Negative", "Positive"]
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Review: {text}\nSentiment: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Review: {text}\nIs the sentiment of the above review Negative or Positive?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

# TODO: add my own dataset
@attr.s(auto_attribs=True)
class YouTube(DataParams):
    dataset: D = D.YOUTUBE
    task: T = T.SENTIMENT
    train_split: str = 'train'
    test_split: str = 'test'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        # TODO parse json files with datasets lib
        return load_dataset('../data/youtube')
    
    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Your task is to classify YouTube videos as Harmful or Harmless based on their metadata.'
        T.prefix_template = instruction if self.prefix else ''
        
        T.example_template = ClassificationTemplate(
            choices = ["Harmful", "Harmless"],
            get_target = lambda _choices, **kwargs: kwargs['label'],
            # templates = 'Title: {Title}\nDescription: {Description}\nTranscript: {Transcript}\nClassification: {_target}')
            templates = 'Title: {Title}\nClassification: {_target}')
        
        T.instructed_example_template = ClassificationTemplate(
            choices = ["Harmful", "Harmless"],
            get_target = lambda _choices, **kwargs: kwargs['label'],
            templates='Review the following video metadata:\nTitle: {Title}\nIs the video above Harmful or Harmless?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class RottenTomatoes(DataParams):
    dataset: D = D.ROTTEN_TOMATOES
    task: T = T.SENTIMENT
    train_split: str = 'train'
    test_split: str = 'validation'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('rotten_tomatoes')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Classify the sentiment of the following reviews into Negative or Positive.'
        T.prefix_template = instruction if self.prefix else ''
        choices=["Negative", "Positive"]
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Review: {text}\nSentiment: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Review: {text}\nIs the sentiment of the above review Negative or Positive?\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

# ---------------------------------------------------------------------------- #
#                             Paraphrase Detection                             #
# ---------------------------------------------------------------------------- #

@attr.s(auto_attribs=True)
class MRPC(DataParams):
    dataset: D = D.MRPC
    task: T = T.PARAPHRASE
    train_split: str = 'train'
    test_split: str = 'validation'
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('glue', 'mrpc', revision='fd8e86499fa5c264fcaad392a8f49ddf58bf4037')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Do these pairs of sentences convey the same meaning? Yes or No.'
        T.prefix_template = instruction if self.prefix else ''
        choices = ['No', 'Yes']
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Sentence 1: {sentence1}\nSentence 2: {sentence2}\nDo the above sentences convey the same meaning? Yes or No.\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class QQP(DataParams):
    dataset: D = D.QQP
    task: T = T.PARAPHRASE
    train_split: str = 'train'
    test_split: str = 'validation'
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        return load_dataset('glue', 'qqp', revision='fd8e86499fa5c264fcaad392a8f49ddf58bf4037')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Are these questions paraphrases of each other? Yes or No.'
        T.prefix_template = instruction if self.prefix else ''
        choices = ['No', 'Yes']
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Question 1: {question1}\nQuestion 2: {question2}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Question 1: {question1}\nQuestion 2: {question2}\nAre Questions 1 and 2 asking the same thing? Yes or No.\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class PAWS(DataParams):
    dataset: D = D.PAWS
    task: T = T.PARAPHRASE
    train_split: str = 'train'
    test_split: str = 'validation'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        return load_dataset('paws', 'labeled_final', revision='cd6b868f3d2d71e9708ed861deee4bbc4d32441e')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Are these sentences paraphrases of each other? Yes or No.'
        T.prefix_template = instruction if self.prefix else ''
        choices = ['No', 'Yes']
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre these sentences paraphrases of each other? Yes or No.\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class PAWSX(DataParams):
    dataset: D = D.PAWSX
    task: T = T.PARAPHRASE
    split: str = 'es'
    train_split: str = 'train'
    test_split: str = 'validation'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        return load_dataset('paws-x', self.split, revision='8a04d940a42cd40658986fdd8e3da561533a3646')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Are these sentences paraphrases of each other? Yes or No.'
        T.prefix_template = instruction if self.prefix else ''
        choices = ['No', 'Yes']
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre the above sentences paraphrases of each other? Yes or No.\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

# ---------------------------------------------------------------------------- #
#                             Commonsense Reasoning                            #
# ---------------------------------------------------------------------------- #

@attr.s(auto_attribs=True)
class COPA(DataParams):
    dataset: D = D.COPA
    task: T = T.COMMONSENSE
    train_split: str = 'train'
    test_split: str = 'validation'
    sel_prompt_version: str = 'v0'
    icL_prompt_version: str = 'v1'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        return load_dataset('super_glue', 'copa')

    def get_templates(self) -> Templates:
        T = Templates()
        T.prefix_template = ''
        if self.icL_prompt_version == 'v0':
            T.example_template = ClassificationTemplate(
                choices=['A', 'B'],
                templates='{premise}\nWhat is the most likely {question} in the above sentence?\nOption A: {choice1}\nOption B: {choice2}\nAnswer: {_target}')
        elif self.icL_prompt_version == 'v1':
            T.example_template = ClassificationTemplate(
                choices=lambda **kwargs: [kwargs['choice1'], kwargs['choice2']],
                templates='{premise}\nWhat is the most likely {question} in the above sentence?\n{_target}')
        else:
            raise ValueError(f'Invalid icL_prompt_version: {self.icL_prompt_version}')

        if self.sel_prompt_version == 'v0':
            T.instructed_example_template = ClassificationTemplate(
                choices=['A', 'B'],
                templates='{premise}\nWhat is the most likely {question} in the above sentence?\nOption A: {choice1}\nOption B: {choice2}\nAnswer: {_target}')
        elif self.sel_prompt_version == 'v1':
            T.instructed_example_template = ClassificationTemplate(
                choices=lambda **kwargs: [kwargs['choice1'], kwargs['choice2']],
                templates='{premise}\nWhat is the most likely {question} in the above sentence: "{choice1}" or "{choice1}"?\nAnswer: {_target}'
            )
        else:
            raise ValueError(f'Invalid selection prompt version: {self.sel_prompt_version}')

        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class Swag(DataParams):
    dataset: D = D.SWAG
    task: T = T.COMMONSENSE
    train_split: str = 'train'
    test_split: str = 'validation'
    is_turbo: bool = False
    icL_prompt_version: str = 'v1'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        return load_dataset('swag')

    def get_templates(self) -> Templates:
        T = Templates()

        if self.icL_prompt_version == 'v0':
            instruction = 'Choose the most likely continuation of the passages.'
            T.prefix_template = instruction if self.prefix else ''
            T.example_template = ClassificationTemplate(
                choices=['A', 'B', 'C', 'D'],
                templates='Passage: {startphrase}\nOption A: {ending0}\nOption B: {ending1}\nOption C: {ending2}\nOption D: {ending3}\nAnswer: {_target}')
        elif self.icL_prompt_version == 'v1':
            instruction = ''
            T.prefix_template = instruction if self.prefix else ''
            T.example_template = ClassificationTemplate(
                choices=lambda **kwargs: [kwargs['ending0'], kwargs['ending1'], kwargs['ending2'], kwargs['ending3']],
                templates='Passage: {startphrase}\nWhat is the most likely continuation of the above passage?\n{_target}')
        else:
            raise ValueError(f'Invalid icL_prompt_version: {self.icL_prompt_version}')
        T.instructed_example_template = ClassificationTemplate(
            choices=['A', 'B', 'C', 'D'],
            templates='What is the most likely continuation of this passage: {startphrase}\nOption A: {ending0}\nOption B: {ending1}\nOption C: {ending2}\nOption D: {ending3}\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class HellaSwag(DataParams):
    dataset: D = D.HELLASWAG
    task: T = T.COMMONSENSE
    train_split: str = 'train'
    test_split: str = 'validation'
    is_turbo: bool = False
    icl_prompt_version: str = 'v1'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        return load_dataset('Rowan/hellaswag')

    def get_templates(self) -> Templates:
        T = Templates()

        def get_options(**kwargs):
            return {chr(ord('A') + i): option.strip()
                    for i, option in enumerate(kwargs['endings'])}
        if self.is_turbo: assert self.icl_prompt_version == 'v0'
        if self.icl_prompt_version == 'v0':
            instruction = 'Choose the option that is the most likely continuation of the passages.'
            T.prefix_template = instruction if self.prefix else ''
            choices = ['A', 'B', 'C', 'D']
            get_target = lambda _choices, **kwargs: chr(ord('A') + int(kwargs['label']))
            def icl_template(test=False, **kwargs):
                template = 'Passage: {ctx}\nOPTIONS:\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer: {_target}'
                if test: template = template[:-len(r'{_target}')]
                return template.format(**get_options(**kwargs), **kwargs)
            T.example_template = ClassificationTemplate(choices=choices, get_target=get_target,
            templates=dict(train=icl_template, test=partial(icl_template, test=True)))
        elif self.icl_prompt_version == 'v1':
            instruction = ''
            T.prefix_template = instruction if self.prefix else ''
            choices = lambda **kwargs: [option.strip() for option in kwargs['endings']]
            get_target = lambda _choices, **kwargs: _choices[int(kwargs['label'])].strip()
            def icl_template(test=False, **kwargs):
                # template = 'Passage: {ctx}\nWhat is the most likely continuation of the above passage?\n{_target}'
                template = '{ctx}\t{_target}'
                # template = 'OPTIONS:\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nFinish the following passage with the most likely continuation from the options above: {ctx} {_target}'
                if test: template = template[:-len(r'{_target}')]
                return template.format(**get_options(**kwargs), **kwargs)
            T.example_template = ClassificationTemplate(choices=choices, get_target=get_target,
            templates=dict(train=icl_template, test=partial(icl_template, test=True)))
        else:
            raise ValueError(f'Invalid icL_prompt_version: {self.icL_prompt_version}')

        def instructed_template(test=False, **kwargs):
            template = 'What is the most likely continuation of the following passage:\nPassage: {ctx}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nAnswer: {_target}'
            if test: template = template[:-len(r'{_target}')]
            return template.format(**get_options(**kwargs), **kwargs)

        T.instructed_example_template = ClassificationTemplate(choices=choices, get_target=get_target,
            templates=dict(train=instructed_template, test=partial(instructed_template, test=True)))
        T.selection_example_template = T.instructed_example_template
        return T
@attr.s(auto_attribs=True)
class PIQA(DataParams):
    dataset: D = D.PIQA
    task: T = T.COMMONSENSE
    train_split: str = 'train'
    test_split: str = 'validation'
    icL_prompt_version: str = 'v2'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data') -> Dataset:
        return load_dataset('piqa')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Select the most logical way to accomplish these goals.'
        T.prefix_template = instruction if self.prefix else ''
        print(self.icL_prompt_version)
        if self.icL_prompt_version == 'v0':
            T.example_template = ClassificationTemplate(
                choices=['A', 'B'],
                templates='Goal: {goal}\nOption A: {sol1}\nOption B: {sol2}\nAnswer: {_target}')
        elif self.icL_prompt_version == 'v1':
            T.prefix_template = ''
            T.example_template = ClassificationTemplate(
                choices=['A', 'B'],
                templates='What is the most logical way to accomplish this goal: {goal}\nOPTIONS:\nA) {sol1}\nB) {sol2}\nAnswer: {_target}')
        elif self.icL_prompt_version == 'v2':
            T.prefix_template = ''
            T.example_template = ClassificationTemplate(
                choices=lambda **kwargs: [kwargs['sol1'], kwargs['sol2']],
                templates='Goal: {goal}\nWhat is the most logical way to accomplish the above goal?\n{_target}')
        else:
            raise ValueError(f'Invalid selection prompt version: {self.icL_prompt_version}')
        T.instructed_example_template = ClassificationTemplate(
            choices = ['A', 'B'],
            get_target = lambda choices, **kwargs: chr(ord('A') + kwargs['label']),
            templates='Goal: {goal}\nWhich of these is the most logical way to accomplish the above goal?: \nOption A: {sol1}\nOption B: {sol2}\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template

        return T

@attr.s(auto_attribs=True)
class CMSQA(DataParams):
    dataset: D = D.CMSQA
    task: T = T.COMMONSENSE
    train_split: str = 'train'
    test_split: str = 'validation'
    prompt_format: str = 'QC-A'      # 'Q-A' or 'QC-A'
    n_input: int = 2

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('commonsense_qa')

    def get_templates(self) -> Templates:
        T = Templates()
        instruction = 'Answer the following multiple-choice questions.'
        T.prefix_template = instruction if self.prefix else ''
        choices = ['A', 'B', 'C', 'D', 'E']
        get_target = lambda _choices, **kwargs: kwargs['answerKey']
        def template(instructed=False, test=False, **kwargs):
            options = {chr(ord('A') + i): option.strip()
                for i, option in enumerate(kwargs['choices']['text'])}
            template = 'Question: {question}\nOption A: {A}\nOption B: {B}\nOption C: {C}\nOption D: {D}\nOption E: {E}\nAnswer: {_target}'
            if instructed:
                template = f'Select one of the choices that best answers the following question:\n{template}'
            if test: template = template[:-len(r'{_target}')]
            return template.format(**options, **kwargs)
        instructed_template = partial(template, instructed=True)
        T.example_template = ClassificationTemplate(choices=choices, get_target=get_target,
            templates=dict(train=template, test=partial(template, test=True)))
        T.instructed_example_template = ClassificationTemplate(choices=choices, get_target=get_target,
            templates=dict(train=instructed_template, test=partial(instructed_template, test=True)))
        T.selection_example_template = T.instructed_example_template
        return T

# ---------------------------------------------------------------------------- #
#                                 Summarization                                #
# ---------------------------------------------------------------------------- #
      
@attr.s(auto_attribs=True)
class AGNews(DataParams):
    dataset: D = D.AGNEWS
    task: T = T.SUMMARIZATION
    train_split: str = 'train'
    test_split: str = 'test'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset(
            path=f'{dataloaders_dir}/classification/agnews.py',
            data_dir=f'{data_root}/classification/agnews')

    def get_templates(self):
        choices = ["World", "Sports", "Business", "Technology"]
        choices_str = ', '.join(choices)
        T = Templates()
        instruction = f'Classify the news articles into one of the following categories: {choices_str}.'
        T.prefix_template = instruction if self.prefix else ''
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Article: {text}\nCategory: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates=f'Classify the following news article into one of these categories: {choices_str}.\n{{text}}\nCategory: {{_target}}')
        T.selection_example_template = T.instructed_example_template
        return T

# ---------------------------------------------------------------------------- #
#                              Question Answering                              #
# ---------------------------------------------------------------------------- #

@attr.s(auto_attribs=True)
class DROP(DataParams):
    dataset: D = D.DROP
    task: T = T.RC
    train_split: str = 'train'
    test_split: str = 'validation'
    embed_context: bool | None = True   # Whether to embed the context.

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset(path=f'{dataloaders_dir}/drop.py')

    def get_templates(self):
        T = Templates()
        instruction = f'Answer the following questions given the corresponding passage.'
        T.prefix_template = instruction if self.prefix else ''
        get_target = lambda **kwargs: kwargs['answer_text']
        T.example_template = GenerationTemplate(get_target=get_target,
            templates='Passage: {passage}\nQuestion: {question}\nAnswer: {_target}')
        T.instructed_example_template = GenerationTemplate(get_target=get_target,
            templates='{passage}\n{question}\nAnswer: {_target}')
        if self.embed_context:
            T.selection_example_template = T.instructed_example_template
        else:
            T.selection_example_template = GenerationTemplate(get_target=get_target,
                templates='{question}\n{_target}')
        return T

@attr.s(auto_attribs=True)
class BoolQ(DataParams):
    dataset: D = D.BOOLQ
    task: T = T.RC
    train_split: str = 'train'
    test_split: str = 'validation'
    embed_context: bool | None = True   # Whether to embed the context.

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('super_glue', 'boolq')

    def get_templates(self):
        T = Templates()
        instruction = f'Answer the following questions given the corresponding passage as yes or no.'
        T.prefix_template = instruction if self.prefix else ''
        choices = ['no', 'yes']
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Passage: {passage}\nQuestion: {question}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='{passage}\n{question} (yes or no)\nAnswer: {_target}')
        if self.embed_context:
            T.selection_example_template = T.instructed_example_template
        else:
            T.selection_example_template = ClassificationTemplate(choices=choices,
                templates='{question}\n{_target}')
        return T


# ---------------------------------------------------------------------------- #
#                                     MISC                                     #
# ---------------------------------------------------------------------------- #

@attr.s(auto_attribs=True)
class CoLA(DataParams):
    dataset: D = D.COLA
    task: T = T.MISC
    train_split: str = 'train'
    test_split: str = 'validation'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('glue', 'cola', revision='fd8e86499fa5c264fcaad392a8f49ddf58bf4037')

    def get_templates(self):
        T = Templates()
        T.prefix_template = "Are the following sentences grammatically correct? (Yes or No)"
        choices = ["No", "Yes"]
        T.example_template = ClassificationTemplate(choices=choices,
            templates='Sentence: {sentence}\nAnswer: {_target}')
        T.instructed_example_template = ClassificationTemplate(choices=choices,
            templates='Is the following sentence grammatical (Yes or No)?\n{sentence}\nAnswer: {_target}')
        T.selection_example_template = T.instructed_example_template
        return T

@attr.s(auto_attribs=True)
class TweetEval(DataParams):
    dataset: D = D.TWEET
    task: T = T.MISC
    split: str = 'emotion'
    train_split: str = 'train'
    test_split: str = 'validation'

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        if self.split != 'stance':
            return load_dataset('tweet_eval', self.split, revision='675a13f04b51b72ccb66915dd5ca2bb13aee8412')
        else:
            stance_targets = ['abortion', 'atheism', 'climate', 'feminist', 'hillary']
            target2name = {'abortion': 'Abortion', 'atheism': 'Atheism', 'climate': 'Climate Change', 'feminist': 'Feminism', 'hillary': 'Hillary Clinton'}
            ds = {'train': [], 'validation': [], 'test': []}
            for t in stance_targets:
                t_ds = load_dataset('tweet_eval', f'stance_{t}', revision='675a13f04b51b72ccb66915dd5ca2bb13aee8412')
                for s in ['train', 'validation', 'test']:
                    ds[s].append(t_ds[s].add_column('target', [target2name[t]] * len(t_ds[s])))
            from datasets import concatenate_datasets, DatasetDict
            return DatasetDict({s: concatenate_datasets(ds[s]) for s in ds})

    def get_templates(self):
        split2iclprefix = {
            'emotion': 'Classify the emotion in the following tweets as one of (A) anger, (B) joy, (C) optimism, or (D) sadness.',
            'sentiment': 'Classifiy the sentiment in the following tweets as one of negative, neutral, or positive.',
            'offensive': 'Are the following tweets offensive? Say Yes or No.',
            'irony': 'Are the following tweets ironic? Say Yes or No.',
            'stance': "Classify the stance of following tweets on the corresponding target as one of (A) favor, (B) against, or (C) neutral.",
            # 'stance_feminist': 'Classify the stance of  as one of favor, against, or neither.',
        }
        split2iclchoices = {
            'emotion': ['A', 'B', 'C', 'D'],
            'sentiment': ['negative', 'neutral', 'positive'],
            'offensive': ['No', 'Yes'],
            'irony': ['No', 'Yes'],
            'stance': ['C', 'B', 'A'],
        }
        split2selprefix = {
            'emotion': 'Classify the emotion in the following tweet as one of anger, joy, optimism, or sadness.',
            'sentiment': 'Classifiy the sentiment in the following tweet as one of negative, neutral, or positive.',
            'offensive': 'Classifiy the following tweet as offensive or non-offensive.',
            'irony': 'Classifiy the following tweet as ironic or non-ironic.',
        }
        split2selchoices = {
            'emotion': ['anger', 'joy', 'optimism', 'sadnesss'],
            'sentiment': ['negative', 'neutral', 'positive'],
            'offensive': ['No', 'Yes'],
            'irony': ['No', 'Yes'],
            'stance': ['neutral', 'against', 'favor']
        }
        T = Templates()
        T.prefix_template = split2iclprefix[self.split] if self.prefix else ''
        if self.split != 'stance':
            T.example_template = ClassificationTemplate(choices=split2iclchoices[self.split],
                templates='Tweet: {text}\nAnswer: {_target}')
            T.instructed_example_template = ClassificationTemplate(choices=split2selchoices[self.split],
                templates=f'{split2selprefix[self.split]}.\nTweet: {{{"text"}}}\nAnswer: {{{"_target"}}}')
        else:
            T.example_template = ClassificationTemplate(choices=split2iclchoices[self.split],
                templates="Tweet: {text}\nTarget: {target}\nAnswer: {_target}")
            T.instructed_example_template = ClassificationTemplate(choices=split2selchoices[self.split],
                templates="Tweet: {text}\nWhat's the tweet's stance on {target}?\nStance: {_target}")
        T.selection_example_template = T.instructed_example_template
        return T

# ---------------------------------------------------------------------------- #
#                                 CoT Reasoning                                #
# ---------------------------------------------------------------------------- #

@attr.s(auto_attribs=True)
class GSM8K(DataParams):
    dataset: D = D.GSM8K
    task: T = T.COT
    train_split: str = 'train'
    test_split: str = 'test'
    sel_prompt_version: str = 'v0'  # 'v0' for bertscore

    def get_dataset(self, data_root: str = '../data', dataloaders_dir: str = 'data'):
        return load_dataset('gsm8k', 'main', revision='25fb0412dbcc2e5128322234d829fbd6ece72c7c')

    def get_templates(self):
        from prompts.base import GSM8KAccuracy
        T = Templates()
        instruction = 'Answer the following question through careful, concise step-by-step reasoning.'
        T.prefix_template = instruction if self.prefix else ''
        common_args = dict(
            get_target='answer',
            metrics = [GSM8KAccuracy()],
            result_metric='accuracy',
        )
        T.example_template = GenerationTemplate(**common_args,
            templates='Question: {question}\nSolution: {_target}')
        T.instructed_example_template = GenerationTemplate(**common_args,
            templates='Answer the following question through careful, concise step-by-step reasoning:\nQuestion: {question}\nSolution: {_target}')
        if self.sel_prompt_version == 'v0':
            T.selection_example_template = GenerationTemplate(**common_args,
                templates='{question}\n{_target}')
        elif self.sel_prompt_version == 'v1':
            T.selection_example_template = T.instructed_example_template
        else:
            T.selection_example_template = GenerationTemplate(**common_args,
                templates='Give the step-by-step reasoning process and then the final answer.{question}\n{_target}')
        return T


all_datasets = [
    GeoQuery, SMCalFlowCS, Atis, Overnight, Break, MTOP, CFQ, COGS, Spider,
    QNLI, MNLI, RTE, WANLI, XNLI, MedNLI,
    SST2, SST5, Yelp, RottenTomatoes,YouTube,
    MRPC, QQP, PAWS, PAWSX,
    COPA, HellaSwag, Swag, PIQA, CMSQA,
    AGNews,
    CoLA, TweetEval,
    DROP, BoolQ,
    GSM8K
]

not_supported = []

ds2cls: dict[D, Type[DataParams]] = {ds_cls().dataset: ds_cls for ds_cls in all_datasets}
# size mismatch w.r.t. https://www.semanticscholar.org/reader/ae22f7c57916562e2729a1a7f34298e4220b77a7: yelp, CommonGen, E2ENLG, DART, AESLC
def  test(
    ds_cls_l: list[Dataset] | Type[DataParams] | list[Type[DataParams]] = 'SMCALFLOW_CS;BREAK;MTOP',
    filter_long_instances: bool = True, rich_output: bool = True,
):
    if isinstance(ds_cls_l[0], Dataset):
        ds_cls_l = [ds2cls[ds] for ds in ds_cls_l]
    if not isinstance(ds_cls_l, list):
        ds_cls_l = [ds_cls_l]

    data_root = Path('../data')
    dataloaders_dir = Path('data')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", use_fast=True) \
        if filter_long_instances else None
    results_l = []
    if rich_output: print = rich.print
    for i, ds_cls in enumerate(ds_cls_l):
        if rich_output:
            print(Rule(title=f' ({i+1}/{len(ds_cls_l)}) {ds_cls.__name__} ', characters='*'))
        else:
            print(f' ({i+1}/{len(ds_cls_l)}) {ds_cls.__name__} '.center(80, '*'))
        if ds_cls in not_supported:
            print(f'{ds_cls.name} is not currently supported')
            continue
        try:
            DP = ds_cls(n_cands=-1, n_test=-1)
            datasets = DP.get_dataset(data_root, dataloaders_dir)
            print(datasets)
            train_ds, candidates, test_ds = DP.get_splits(
                data_root, dataloaders_dir, tokenizer)
            print(candidates)
            print(test_ds)
            print(test_ds[0])
            if rich_output:
                print(Rule(title='Templates', characters='-'))
            else:
                print(' Templates '.center(80, '-'))
            T = DP.get_templates()
            ex = test_ds[0]
            print(T.prefix_template)

            print("ICL Example Template:")
            print(T.example_template.format(**ex))
            print()
            # print(T.instructed_example_template.format(**ex))

            print("Selection Example Template:")
            print(T.selection_example_template.format(**ex, test=True))
            print()

            if rich_output:
                print(Rule(title='Metrics', characters='-'))
            else:
                print(' Metrics '.center(80, '-'))
            ex_template = T.example_template
            if not hasattr(ex_template, 'get_choices'):
                pred = ex_template.get_target(**ex)
                metrics = ex_template.check_output(pred, _target=pred, **ex)
            else:
                pred = ex_template.get_target(ex_template.get_choices(**ex), **ex)
                metrics = ex_template.check_output(pred, **ex)
            print(f'Prediction: {pred}')
            print(f'Metrics: {metrics}')
            res = dict(
                dataset=DP.dataset,
                task_category=DP.task,
                ds_sizes=(len(train_ds), len(test_ds)),
                prefix=T.prefix_template,
                example=T.example_template.format(**ex),
                instructed_example=T.instructed_example_template.format(**ex),
                selection_example_template=T.selection_example_template.format(**ex, test=True),
                metric=T.example_template.result_metric,
            )
            results_l.append(res)
        except Exception as e:
            print(f'{ds_cls.__name__} failed')
            raise e
        print()
    import pandas as pd
    resultsdf = pd.DataFrame(results_l).sort_values(by=['task_category', 'dataset'])
    return resultsdf

@app.command()
def make_prompts_table(datasets: str = 'SMCALFLOW_CS;BREAK;MTOP'):
    from pathlib import Path
    data_root = Path('../data')
    dataloaders_dir = Path('data')
    ds_cls_l = [ds2cls[ds] for ds in get_datasets(datasets)]
    records = []
    latex_table_records = []
    for i, ds_cls in enumerate(ds_cls_l):
        print(f' ({i+1}/{len(ds_cls_l)}) {ds_cls.__name__} '.center(80, '*'))
        DP = ds_cls(n_cands=-1, n_test=-1)
        datasets = DP.get_dataset(data_root, dataloaders_dir)
        train_ds, candidates, test_ds = DP.get_splits(data_root, dataloaders_dir)
        ex = test_ds[0]
        T = DP.get_templates()
        icl_template = T.example_template.format(**ex)
        sel_template = T.selection_example_template.format(**ex, test=True)
        records.append({
            'Task Category': DP.task,
            'Dataset': DP.dataset,
            'ICL Example Template': icl_template,
            'Selection Example Template': sel_template,
        })
        Path(f'../assets/prompts/icl/{DP.dataset.name}.txt').write_text(icl_template)
        Path(f'../assets/prompts/sel/{DP.dataset.name}.txt').write_text(sel_template)
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_excel('prompts.xlsx')
    return df

datasets_str = []
@app.command()
def test_datasets(datasets: str = 'SMCALFLOW_CS;BREAK;MTOP', filter_long_instances: bool = True, rich_output: bool = True,):
    """
    Example Usage: python data_params.py --datasets "SMCALFLOW_CS;BREAK;MTOP"

    datasets: list of names from constants.Dataset as a ';' separated string
    """
    ds_cls_l = [ds2cls[ds] for ds in get_datasets(datasets)]
    test(ds_cls_l, filter_long_instances, rich_output)

if __name__ == '__main__':
    app()