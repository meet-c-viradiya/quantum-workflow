from setuptools import setup, find_packages

setup(
    name='quantum-workflow',
    version='0.1.0',
    author='Your Name',
    author_email='your-email@example.com',
    description='A package for scheduling and optimizing quantum workflows.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/quantum-workflow',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'networkx>=2.6.0',
        'matplotlib>=3.4.0',
        'qiskit>=1.0.0',
        'qiskit-algorithms>=0.2.0',
        'qiskit-optimization>=0.5.0',
        'qiskit-aer>=0.12.0',
        'scipy>=1.7.0',
        'seaborn>=0.11.0',
        'pygraphviz>=1.7',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)