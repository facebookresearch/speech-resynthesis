# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json


def append_to_config(config_file, attributes):
    """Add to the config file new attributes
    (without changing the order of old config)
    Args:
        - config_file (str): path to config file
        - attributes (list): list of tuple ('att_name', att_value)
    """
    if isinstance(attributes, tuple):
        assert len(attributes) == 2
        append_to_config(config_file, [attributes])
        return

    # Read the config lines
    with open(config_file) as f:
        data = f.read()
    cfg_lines = data.split("\n")

    # This is to deal with empty lines
    nnempty_ids = [i for i, item in enumerate(cfg_lines) if item != ""]
    assert cfg_lines[nnempty_ids[0]].rstrip() == "{", cfg_lines[nnempty_ids[0]]
    assert cfg_lines[nnempty_ids[-1]].rstrip() == "}", cfg_lines[nnempty_ids[-1]]

    # Read the config to dict (to verify & check existing attributes later)
    cfg = json.loads(data)

    # Add a , to the last item
    if cfg_lines[nnempty_ids[-2]].rstrip()[-1] != ",":
        cfg_lines[nnempty_ids[-2]] = cfg_lines[nnempty_ids[-2]].rstrip() + ","

    # Add the attributes to lines
    cursor_id = nnempty_ids[-2]
    has_added = False
    for att_name, att_value in attributes:
        if att_name in cfg:
            print(
                f'Attempted to add existing attribute "{att_name}" to {config_file}, skipping...'
            )
            continue
        if isinstance(att_value, str):
            cfg_line = f'    "{att_name}": "{att_value}",'
        else:
            cfg_line = f'    "{att_name}": {att_value},'
        cursor_id += 1
        cfg_lines.insert(cursor_id, cfg_line)
        has_added = True

    if not has_added:
        return

    # Remove the last trailing ,
    cfg_lines[cursor_id] = cfg_lines[cursor_id].rstrip(",")

    # Write the output
    with open(config_file, "w") as f:
        f.write("\n".join(cfg_lines))
