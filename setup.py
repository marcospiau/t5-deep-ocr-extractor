from setuptools import find_packages, setup

setup(
    name='final_project_ia376j',
    packages=find_packages(),
    version='0.1.0',
    description='Final Projetct IA376J - SROIE Task 3`',
    author='marcospiau',
    license='MIT',
    package_data={
        '': ['*.gin'],
    },
    install_requires = [
        'absl-py==0.11.0',
        'fairseq==0.10.1',
        'gin-config==0.4.0',
        'neptune-client==0.4.130',
        'pytorch-lightning==1.1.0',
        'sentencepiece==0.1.91',
        'tokenizers==0.9.4',
        'torch==1.4.0',
        'transformers==4.0.1'
    ]
)
