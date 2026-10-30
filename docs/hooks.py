"""MkDocs hook: publish files that live outside docs/ without copying them.

The example notebooks stay in examples/ (where users find them on GitHub) and
the Northflow mark stays in .github/assets/; this hook adds them to the site
before mkdocs-jupyter converts the notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mkdocs.plugins import event_priority
from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parent.parent
# (path inside the site, source directory): the file is read from source/path.
EXTERNAL = [("assets/northflow-mark.svg", ROOT / ".github")]


@event_priority(100)  # before mkdocs-jupyter looks at the notebooks
def on_files(files: Files, config: Any) -> Files:
    sources = list(EXTERNAL)
    sources += [(f"examples/{nb.name}", ROOT) for nb in sorted((ROOT / "examples").glob("*.ipynb"))]
    for uri, src_dir in sources:
        if files.get_file_from_path(uri) is None:
            files.append(
                File(
                    uri,
                    src_dir=str(src_dir),
                    dest_dir=config["site_dir"],
                    use_directory_urls=config["use_directory_urls"],
                )
            )
    return files
