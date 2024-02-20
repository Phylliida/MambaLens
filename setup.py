import setuptools

setuptools.setup(
    name = "mamba_lens",
    version = "0.0.2",
    author = "Phylliida",
    author_email = "phylliidadev@gmail.com",
    description = "TransformerLens port for Mamba",
    url = "https://github.com/Phylliida/MambaLens.git",
    project_urls = {
        "Bug Tracker": "https://github.com/Phylliida/MambaLens/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    python_requires = ">=3.6",
    install_requires = ['transformer-lens', 'torch', 'einops', 'jaxtyping']
)
