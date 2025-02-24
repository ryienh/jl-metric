from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jl-metric",
    version="0.1.0",
    author="Ryien Hosseini",
    author_email="ryien@uchicago.edu",
    description="A Johnson-Lindenstrauss-based metric for dynamic graph generative model evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryienh/jl-metric",
    project_urls={
        "Bug Tracker": "https://github.com/ryienh/jl-metric/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "numpy>=1.19.0",
    ],
    keywords="graph-learning, dynamic-graphs, generative-models",
)
