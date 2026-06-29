from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "authority_data" / "data"
DEFAULT_SOURCE_DIR = REPO_ROOT / "authority_data" / "sources"
UPSTREAM_REPOS = {
    "agentdojo": "https://github.com/ethz-spylab/agentdojo.git",
    "injecagent": "https://github.com/uiuc-kang-lab/InjecAgent.git",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from authority_data.authority import export_authority_jsonl
from authority_data.benchmarks import export_agentdojo_jsonl, export_injecagent_jsonl


def _optional_limit(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def _run_git(args: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        command = " ".join(["git", *args])
        raise RuntimeError(f"Failed to run {command}")


def ensure_repo(name: str, root: Path, *, fetch: bool = True) -> Path:
    root = root.expanduser().resolve()
    repo_url = UPSTREAM_REPOS[name]
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", "--depth", "1", repo_url, str(root)])
    elif (root / ".git").exists() and fetch:
        _run_git(["pull", "--ff-only"], cwd=root)
    elif not (root / ".git").exists():
        raise FileNotFoundError(f"{root} exists but is not a git clone")
    return root


def make_authority_data(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    make_authority: bool = True,
    make_agentdojo: bool = True,
    make_injecagent: bool = True,
    agentdojo_root: Path | None = None,
    injecagent_root: Path | None = None,
    fetch: bool = True,
    max_authorities: int = 20,
    max_situations_per_authority: int = 5,
    max_noise: int = 5,
    max_agentdojo_records: int | None = 20,
    max_injecagent_examples: int | None = None,
) -> dict[str, dict[str, int]]:
    data_dir = data_dir.expanduser().resolve()
    source_dir = source_dir.expanduser().resolve()
    results: dict[str, dict[str, int]] = {}

    if make_authority:
        results["authority"] = export_authority_jsonl(
            output_dir=data_dir / "authority",
            max_authorities=max_authorities,
            max_situations_per_authority=max_situations_per_authority,
            max_noise=max_noise,
        )

    if make_agentdojo:
        resolved_agentdojo_root = ensure_repo(
            "agentdojo",
            agentdojo_root or source_dir / "agentdojo",
            fetch=fetch,
        )
        results["agentdojo"] = export_agentdojo_jsonl(
            source_root=resolved_agentdojo_root,
            output_dir=data_dir / "benchmarks" / "agentdojo",
            max_records=max_agentdojo_records,
        )

    if make_injecagent:
        resolved_injecagent_root = ensure_repo(
            "injecagent",
            injecagent_root or source_dir / "injecagent",
            fetch=fetch,
        )
        results["injecagent"] = export_injecagent_jsonl(
            source_root=resolved_injecagent_root,
            output_dir=data_dir / "benchmarks" / "injecagent",
            max_examples=max_injecagent_examples,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Make local authority_data JSONL files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--agentdojo-root", type=Path, default=None)
    parser.add_argument("--injecagent-root", type=Path, default=None)
    parser.add_argument("--skip-authority", action="store_true")
    parser.add_argument("--skip-agentdojo", action="store_true")
    parser.add_argument("--skip-injecagent", action="store_true")
    parser.add_argument("--no-fetch", action="store_true", help="Do not pull existing source clones.")
    parser.add_argument("--max-authorities", type=int, default=20)
    parser.add_argument("--max-situations-per-authority", type=int, default=5)
    parser.add_argument("--max-noise", type=int, default=5)
    parser.add_argument("--max-agentdojo-records", type=int, default=20)
    parser.add_argument("--max-injecagent-examples", type=int, default=-1)
    args = parser.parse_args()

    results = make_authority_data(
        data_dir=args.data_dir,
        source_dir=args.source_dir,
        make_authority=not args.skip_authority,
        make_agentdojo=not args.skip_agentdojo,
        make_injecagent=not args.skip_injecagent,
        agentdojo_root=args.agentdojo_root,
        injecagent_root=args.injecagent_root,
        fetch=not args.no_fetch,
        max_authorities=args.max_authorities,
        max_situations_per_authority=args.max_situations_per_authority,
        max_noise=args.max_noise,
        max_agentdojo_records=_optional_limit(args.max_agentdojo_records),
        max_injecagent_examples=_optional_limit(args.max_injecagent_examples),
    )
    for dataset_name, counts in results.items():
        for split_name, count in counts.items():
            print(f"{dataset_name}/{split_name}: {count}")


if __name__ == "__main__":
    main()
