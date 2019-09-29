import setuptools

with open('requirements.txt', 'r') as fh:
    install_requirements = fh.readlines()

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="cloudbusting",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
#    author="Example Author",
#    author_email="author@example.com",
    description="Understanding cloud organization",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Team-Cloudbusters/cloudbusting',
    package_dir={'': 'lib'},
    packages=setuptools.find_packages('lib'),
    install_requires=install_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        #"Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

