from setuptools import setup, find_packages

setup(
    name="sovits_tokenizer",
    version="0.0.1",
    author="Joseph Cheng",
    author_email="joseph.cheng@hon9kon9ize.com",
    description="SovitsTokenizer: A low-bitrate audio tokenizer that converts speech into discrete tokens",
    long_description=open("README.md").read(),
    license="Apache2.0",
    long_description_content_type="text/markdown",
    url="https://github.com/hon9kon9ize/sovits-tokenizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    entry_points={
        "console_scripts": ["s3tokenizer=s3tokenizer.cli:main"],
    },
    install_requires=[
        "torch>=1.7.0",
        "numpy==1.23.4",
        "librosa==0.9.2",
        "numba==0.56.4",
        "pytorch-lightning",
        "ffmpeg-python",
        "transformers",
        "einops",
    ],
)
