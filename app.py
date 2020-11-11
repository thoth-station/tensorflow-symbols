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

import daiquiri
from pathlib import Path
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
import ast
import glob
import json
import logging
import os
import sys

import attr
import click

from thoth.python import Source


daiquiri.setup(level=logging.INFO)
_LOGGER = daiquiri.getLogger(__name__)


@attr.s(slots=True)
class ImportVisitor(ast.NodeVisitor):
    """Visitor for capturing imports, nodes and relevant parts to be reported."""

    imports = attr.ib(type=dict, default=attr.Factory(dict))
    imports_from = attr.ib(type=dict, default=attr.Factory(dict))

    def visit_Import(self, import_node: ast.Import) -> None:
        """Visit `import` statements and capture imported modules/names."""
        for alias in import_node.names:
            if alias.asname is not None:
                if alias.asname in self.imports:
                    _LOGGER.warning(
                        "Detected multiple imports with same name %r, results of calls "
                        "will differ based on actual execution",
                        alias.asname,
                    )

                self.imports[alias.asname] = alias.name
            else:
                self.imports[alias.name] = alias.name

    def visit_ImportFrom(self, import_from_node: ast.ImportFrom) -> None:
        """Visit `import from` statements and capture imported modules/names."""
        for alias in import_from_node.names:
            if alias.asname:
                if alias.asname in self.imports_from:
                    _LOGGER.warning(
                        "Multiple imports for %r found, detection might give misleading results", alias.asname,
                    )
                self.imports_from[alias.asname] = {
                    "module": import_from_node.module,
                    "name": alias.name,
                }
            else:
                if alias.name in self.imports_from:
                    _LOGGER.warning(
                        "Multiple imports for name %r found, detection might give misleading results", alias.name,
                    )
                self.imports_from[alias.name] = {
                    "module": import_from_node.module,
                    "name": alias.name,
                }


def _iter_python_file_ast(path: str, *, ignore_errors: bool) -> Generator[Tuple[Path, object], None, None]:
    """Get AST for all the files given the path."""
    for python_file in _get_python_files(path):
        python_file_path = Path(python_file)
        _LOGGER.debug("Parsing file %r", str(python_file_path.absolute()))
        try:
            yield python_file_path, ast.parse(python_file_path.read_text())
        except Exception:
            if ignore_errors:
                _LOGGER.exception("Failed to parse Python file %r", python_file)
                continue

            raise


def _get_python_files(path: str) -> List[str]:
    """Get Python files for the given path."""
    if os.path.isfile(path):
        files = [path]
    else:
        files = glob.glob(f"{path}/**/*.py", recursive=True)

    if not files:
        raise FileNotFoundError(f"No files to process for {str(path)!r}")

    return files


@click.group()
def cli() -> None:
    """Extract and manipulate with TensorFlow API."""


@cli.command()
@click.option(
    "--tensorflow-name",
    help="Name of TensorFlow package to be used (e.g. tensorflow, intel-tensorflow, ...).",
    type=str,
    required=False,
    metavar="TF",
    default="tensorflow",
    show_default=True,
)
@click.option(
    "--tensorflow-versions",
    help="TensorFlow versions for which the API should be gathered (comma separated).",
    type=str,
    metavar="TF_VERSION",
    required=False,
    show_default=True,
)
@click.option(
    "--index-url",
    help="Python package index to be used.",
    type=str,
    metavar="URL",
    required=False,
    default="https://pypi.org/simple",
    show_default=True,
)
def gather(tensorflow_name: str, tensorflow_versions: Optional[str], index_url: str) -> None:
    """Gather symbols available in a TensorFlow release."""
    index = Source(index_url)
    tf_versions = tensorflow_versions.split(",") if tensorflow_versions else index.get_package_versions(tensorflow_name)

    os.makedirs("data", exist_ok=True)

    for tf_version in tf_versions:
        # We consider only manylinux releases.
        artifacts = [
            a for a in index.get_package_artifacts(tensorflow_name, tf_version) if "manylinux" in a.artifact_name
        ]
        if not artifacts:
            _LOGGER.error(
                "No manylinux artifact found for %r in version %r on %r", tensorflow_name, tf_version, index.url
            )

        artifact = artifacts[0]
        _LOGGER.info("Downloading artifact %r", artifact.artifact_name)

        artifact._extract_py_module()
        api_path = os.path.join(artifact.dir_name, "tensorflow", "_api")

        file_symbols: Dict[str, List[str]] = {}
        for python_file, file_ast in _iter_python_file_ast(api_path, ignore_errors=True):
            visitor = ImportVisitor()
            visitor.visit(file_ast)  # type: ignore

            file_symbols[str(python_file)] = [
                i for i in (*visitor.imports.keys(), *visitor.imports_from.keys()) if not i.startswith("_")
            ]

        result = []
        for file_name, symbols in file_symbols.items():
            module_name = f"tensorflow.{file_name[file_name.index('/_api/') + 6:]}"
            if module_name.endswith(("__init__.py", ".__init__.py")):
                module_name = module_name[: -len("/__init__.py")]

            if module_name.endswith(".py"):
                module_name = module_name[: -len(".py")]
            module_name = module_name.replace("/", ".")

            # XXX: tensorflow.v2.v2 has some kind of special meaning?!
            module_name = module_name.replace(".v2.v2", ".v2")
            module_name = module_name.replace("tensorflow.v2.", "tensorflow.")

            for symbol in symbols:
                result.append(f"{module_name}.{symbol}")

        with open(os.path.join("data", f"{tf_version}.json"), "w") as f:
            json.dump(result, f, indent=2)


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
