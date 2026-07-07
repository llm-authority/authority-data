from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


def build_setup_kwargs() -> dict[str, object]:
    return {
        "name": "authority-data",
        "version": "0.1.0",
        "description": "Dataset utilities and JSONL files for authority-decision experiments.",
        "long_description": README.read_text(encoding="utf-8") if README.exists() else "",
        "long_description_content_type": "text/markdown",
        "license": "Apache-2.0",
        "license_files": ["LICENSE"],
        "packages": find_packages(include=["src", "src.*"]),
        "py_modules": ["make_data", "hf_push"],
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
                "authority-data-make=make_data:main",
                "authority-data-push-hf=hf_push:main",
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
