from setuptools import setup, find_packages

setup(
    name='adi_pipeline',
    version='0.1',
    description='Basic pipeline for ADI and other ADI tools',
    url='https://github.com/jmilou/adi_pipeline',
    author='Julien Milli',
    author_email='jmilli@eso.org',
    license='MIT',
    keywords='image processing data analysis',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib','pandas','datetime'
    ],
    zip_safe=False
)
