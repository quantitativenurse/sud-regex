from pathlib import Path
from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="sud-regex",  # distribution name on PyPI
    version="0.1.0",
    description="Regex-driven extraction with negation for clinical text (SUD-focused).",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Quantitative Nurse Lab",
    author_email="quantitativenurse@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.21",
    ],
    extras_require={
        "dev": [
            "black==25.1.0",
            "flake8==7.3.0",
            "isort==6.0.1",
            "pytest",
            "build",
            "twine",
        ],
        "viz": ["matplotlib>=3.6"],
        "yaml": ["pyyaml>=6"],
        # convenience bundle:
        "all": ["matplotlib>=3.6", "pyyaml>=6"],
    },
    entry_points={
        "console_scripts": [
            "sudregex=SUDRegex.cli:main",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/quantitativenurse/sud-regex",
        "Issues": "https://github.com/quantitativenurse/sud-regex/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License : MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
    ],
)
