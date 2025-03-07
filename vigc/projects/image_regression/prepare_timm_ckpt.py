"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import timm
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dst-path", required=True)
    args = parser.parse_args()

    return args


def main():
    cfg = parse_args()
    model = timm.create_model(cfg.model_name, pretrained=True)
    state_dict = model.state_dict()
    torch.save(state_dict, cfg.dst_path)


if __name__ == "__main__":
    main()
