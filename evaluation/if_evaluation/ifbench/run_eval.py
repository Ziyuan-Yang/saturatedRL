# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from ifbench_addon.evaluation_lib import *

def if_score(data, responses):
  inputs, prompt_to_response = read_data_and_responses(data, responses)

  outputs_loose = [
      test_instruction_following_loose(inp, prompt_to_response)
      for inp in inputs
  ]
  return [o.follow_all_instructions for o in outputs_loose]
