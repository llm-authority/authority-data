from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"
PACKAGE_ROOT = ROOT / "authority_data"


def _data_files() -> list[str]:
    data_root = PACKAGE_ROOT / "data"
    if not data_root.exists():
        return []
    return [
        str(path.relative_to(PACKAGE_ROOT))
        for path in sorted(data_root.rglob("*.jsonl"))
    ]


def build_setup_kwargs() -> dict[str, object]:
    return {
        "name": "authority-data",
        "version": "0.1.0",
        "description": "Dataset utilities and JSONL files for authority-decision experiments.",
        "long_description": README.read_text(encoding="utf-8") if README.exists() else "",
        "long_description_content_type": "text/markdown",
        "license": "Apache-2.0",
        "license_files": ["LICENSE"],
        "packages": find_packages(include=["authority_data", "authority_data.*", "scripts", "scripts.*"]),
        "package_data": {"authority_data": _data_files()},
        "include_package_data": False,
        "zip_safe": False,
        "python_requires": ">=3.10",
        "install_requires": [],
        "extras_require": {
            "hf": ["datasets>=2.14.0"],
            "notebooks": ["datasets>=2.14.0", "jupyter>=1.0.0"],
            "dev": ["build>=1.0.0", "twine>=4.0.0"],
        },
        "entry_points": {
            "console_scripts": [
                "authority-data-make=scripts.make_authority_data:main",
                "authority-data-export-authority=authority_data.authority:main",
                "authority-data-export-benchmarks=authority_data.benchmarks:main",
                "authority-data-push-hf=scripts.hf_push:main",
            ],
        },
        "classifiers": [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    }


if __name__ == "__main__":
    setup(**build_setup_kwargs())
