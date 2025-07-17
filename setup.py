from setuptools import find_packages, setup

setup(
    name="SUDRegex",  # Your PyPI package name
    version="0.1.0",
    description="Efficient regex and negation detection pipeline for large-scale clinical text analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Christy Parker, Dagmawi Negesse",
    author_email="dagmawi.z.negesse@vumc.org",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "black==25.1.0",
            "flake8==7.3.0",
            "isort==6.0.1",
            "pytest",
        ],
        "parallel": ["pandarallel"],
    },
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "sudregex=SUDRegex.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
