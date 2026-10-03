from setuptools import setup, find_packages

setup(
    name="sutraopt",
    version="1.0.0",
    description="SutraOpt (BharatOpt) — Indigenous GPU-Accelerated Mathematical Optimization Solver",
    packages=find_packages(include=["sutraopt*"]),
    py_modules=["sutra_cli"],
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x>=12.0"],
        "benchmark": ["highspy>=1.5"],
    },
    entry_points={
        "console_scripts": [
            "sutraopt=sutra_cli:main",
        ],
    },
)
