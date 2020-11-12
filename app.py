#!/usr/bin/env python3
# Copyright(C) 2020 Fridolin Pokorny
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""A command line interface to gather and pre-preprocess symbols available in TensorFlow API."""

from collections import deque
from typing import Optional
import daiquiri
import json
import logging
import os
import sys

import click


daiquiri.setup(level=logging.INFO)
_LOGGER = daiquiri.getLogger(__name__)


@click.group()
def cli() -> None:
    """Extract and manipulate with TensorFlow API."""


@cli.command()
def gather() -> None:
    """Gather symbols available in a TensorFlow release."""
    import tensorflow as tf
    import tensorflow._api.v2.v2 as tf_v2

    queue = deque([tf._api.v2, tf_v2])

    result = []
    modules_seen = set()
    while queue:
        module = queue.pop()

        modules_seen.add(module.__name__)

        for item in dir(module):
            obj = getattr(module, item)
            if isinstance(obj, tf.__class__):
                if obj.__name__ not in modules_seen and obj.__name__.startswith("tensorflow._api.v2."):
                    queue.append(obj)
            else:
                m = module.__name__.replace("tensorflow._api.v2", "tensorflow")
                result.append(f"{m}.{item}")

    with open(f"data/{tf.__version__}.json", "w") as f:
        json.dump(sorted(result), f, indent=2)


@cli.command()
@click.option(
    "--path",
    help="A path to gathered files.",
    type=str,
    metavar="DIR",
    default="data",
    required=False,
    show_default=True,
)
@click.option(
    "--no-patch",
    help="Discard patch release information.",
    is_flag=True,
    default=False,
    required=False,
    show_default=True,
)
def merge(path: str, no_patch: bool) -> None:
    """Merge multiple API data files into one."""
    result = {}
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if not os.path.isfile(file_path) or not file_path.endswith(".json"):
            _LOGGER.debug("Skipping file %r", f)
            continue

        key = os.path.basename(f).rsplit(".", maxsplit=1)[0]
        if no_patch:
            # We rely on semver as used by TensorFlow packages.
            key = key.rsplit(".", maxsplit=1)[0]

        if key in result:
            # No additional clever logic is done.
            _LOGGER.warning("Multiple versions for %r detected", key)
            continue

        with open(file_path, "r") as input_file:
            result[key] = json.load(input_file)

    json.dump(result, sys.stdout, indent=2)


__name__ == "__main__" and cli()
