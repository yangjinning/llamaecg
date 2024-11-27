"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
import numpy as np
from serialize import serialize_arr, SerializerSettings

# settings = SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
settings = SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True)


class Prompter(object):
    __slots__ = ("_path_template", "template", "_verbose")

    # __slots__ 是一个特殊的类属性，用于限制类实例只能拥有特定的属性，从而节省内存并提高访问速度。在您提供的 Prompter 类中，__slots__ 被定义为 ("template", "_verbose")，这意味着该类的实例只能包含 template 和 _verbose 这两个属性。

    def __init__(self, path_template: str = "", verbose: bool = False):
        self._verbose = verbose
        if not osp.exists(path_template):
            raise ValueError(f"Can't read {path_template}")
        with open(path_template) as fp:
            self.template = json.load(fp)
        if self._verbose:  # if print the result
            print(
                f"Using prompt template {path_template}: {self.template['description']}"
            )

    # def np2str(self,data,separator=',',precision=4):
    #     data = np.round(data, precision)
    #     return np.array2string(data, separator=separator, formatter={'float_kind':lambda x: f"{x:.{precision}f}"})

    def np2str(self, data, precision=4):
        data = np.round(data, precision)
        return serialize_arr(data, settings)

    def generate_prompt(
            self,
            ecg_data: np.array,
            question: str,
    ) -> str:
        res = self.template["prompt"].format(ecg_data=self.np2str(ecg_data), question=question)
        if self._verbose:
            print(res)
        return res

    def generate_chat_prompt(
            self,
            ecg_data: np.array,
            question: str,
    ) -> str:
        message_text = self.template["prompt"].format(ecg_data=self.np2str(ecg_data), question=question)
        return message_text

    def get_response(self, output: str, use_chat_prompt=False) -> str:
        if use_chat_prompt:
            return output.split(self.template["response_chat_split"])[-1].strip()
        else:
            return output.split(self.template["response_prompt_split"])[1].strip()
