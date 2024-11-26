from setuptools import setup, find_packages

setup(
    name="SovitsTokenizer",
    version="0.0.1",
    author="Your Name",
    author_email="joseph.cheng@hon9kon9ize.com",
    description="A module for tokenizing speech data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hon9kon9ize/sovits-tokenizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "numpy==1.23.4",
        "librosa==0.9.2",
        "numba==0.56.4",
        "pytorch-lightning",
        "ffmpeg-python",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "sovits_tokenizer=sovits_tokenizer.SovitsTokenizer:main",
        ],
    },
)
