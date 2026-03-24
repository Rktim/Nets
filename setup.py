from setuptools import setup, find_packages

setup(
    name="nets",
    version="0.1.0",
    description="Lightweight deep learning framework",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "plotly",
        "dash",
        "scikit-learn"
    ],
)