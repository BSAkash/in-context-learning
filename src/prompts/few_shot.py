"""Prompt template that contains few shot examples."""
import attr
from typing import Any, Union

from langchain.prompts.example_selector.base import BaseExampleSelector
from prompts.base import ExampleTemplate

@attr.s(auto_attribs=True)
class FewShotPromptTemplate2:
    """Prompt template that contains few shot examples."""

    example_template: ExampleTemplate
    """PromptTemplate used to format an individual example."""

    prefix_template: str = ''
    """A prompt template string to put before the examples."""

    suffix_template: str = ''
    """A prompt template string to put after the examples."""

    examples: list[dict] | None = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: BaseExampleSelector | None = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    max_len: int = -1
    subtract_gen_len: bool = False
    enc_len_fn: Any = None

    lm: str = 'EleutherAI/gpt-neo-2.7B'

    def __attrs_post_init__(self):
        """Post init hook to check that the template is valid."""
        self.check_examples_and_selector()


    def check_examples_and_selector(self):
        """Check that one and only one of examples/example_selector are provided."""
        if self.examples and self.example_selector:
            raise ValueError(
                "Only one of 'examples' and 'example_selector' should be provided"
            )

        if self.examples is None and self.example_selector is None:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )

    def _get_examples(self, **kwargs: Any) -> list[dict]:
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        else:
            raise ValueError

    def make_prompt(self, prefix, example_strings, test_example_string):
        pieces = [
            prefix,
            *example_strings,
            test_example_string,
        ]
        # Create the overall prompt.
        return self.example_separator.join([p for p in pieces if p])

    def make_turbo_prompt(self, prefix, examples, test_example_string):
        ET = self.example_template
        if hasattr(self.example_template, 'get_choices'): # classification
            get_target = lambda **ex: ET.get_target(ET.get_choices(**ex), **ex)
        else:
            get_target = ET.get_target
        messages = []
        if prefix: messages.append({"role": "user", "content": prefix})
        for ex in examples:
            messages += [
                {"role": "user", "content": ET.format(**ex, test=True).strip()},
                {"role": "assistant", "content": get_target(**ex)},
            ]
        messages.append({"role": "user", "content": test_example_string.strip()})
        return messages


    def format_from_examples(self, examples, return_demos=False, is_turbo=False, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            examples: A list of exemplars.
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        # Format the examples.
        ET = self.example_template
        example_strings = [ET.format(**ex, test=False) for ex in examples]
        test_example_string = ET.format(**kwargs, test=True)
        # Format the template with the input variables.
        prefix, suffix = self.prefix_template, self.suffix_template
        max_len = self.max_len
        n_shots = len(example_strings)

        if max_len != -1:
            # prune the demonstrations to make sure the prompt will fit in the context length limit
            if not self.subtract_gen_len:
                # print("[LOG] max_len", max_len)
                # print("[LOG] n_shots", n_shots)
                while self.enc_len_fn(self.make_prompt(
                    prefix, example_strings[-n_shots:], test_example_string)
                ) > max_len and n_shots > 1:
                    # example_strings = example_strings[1:]
                    n_shots -= 1
            else:
                test_example_string_completed = ET.format(**kwargs)
                while self.enc_len_fn(self.make_prompt(
                    prefix, example_strings[-n_shots:], test_example_string_completed)
                ) > max_len and n_shots > 1:
                    # example_strings = example_strings[1:]
                    n_shots -= 1
            print(f'reduced examples from {len(examples)} to {len(example_strings)}')

        if not is_turbo:
            prompt = self.make_prompt(prefix, example_strings[-n_shots:], test_example_string)
        else:
            prompt = self.make_turbo_prompt(prefix, list(examples)[-n_shots:], test_example_string)
        if return_demos:
            return prompt, list(examples)[-n_shots:]
        else:
            return prompt

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        return self.format_from_examples(examples, **kwargs)

    def _prompt_dict(self) -> dict:
        """Return a dictionary of the prompt."""
        if self.example_selector:
            raise ValueError("Saving an example selector is not currently supported")

        prompt_dict = self.dict()
        prompt_dict["_type"] = "few_shot"
        return prompt_dict

    def parse_output(self, lm_output: str, **kwargs) -> Union[str, list[str], dict[str, str]]:
        if hasattr(self.example_template, 'parse_output'):
            return self.example_template.parse_output(lm_output, **kwargs)
        else:
            return super().parse_output(lm_output, **kwargs)

    def check_output(self, prediction, **kwargs) -> bool:
        if hasattr(self.example_template, 'check_output'):
            return self.example_template.check_output(prediction, **kwargs)
        else:
            super().check_output(prediction, **kwargs)