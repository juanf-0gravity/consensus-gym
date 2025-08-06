from setuptools import setup, find_packages

setup(
    name="distributed-consensus-gym",
    version="0.1.0",
    author="Juan Flores",
    author_email="rajdesai58.work@gmail.com",
    description="Multi-agent reinforcement learning environment for distributed consensus protocols",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/juanf-0gravity/distributed-consensus-gym",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gym>=0.18.0",
        "numpy>=1.19.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
        "pettingzoo>=1.8.0",
        "ray[rllib]>=1.4.0",
        "scipy>=1.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "consensus-train=consensus_gym.scripts.train:main",
            "consensus-eval=consensus_gym.scripts.evaluate:main",
            "consensus-viz=consensus_gym.scripts.visualize:main",
        ],
    },
)