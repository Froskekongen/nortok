from setuptools import setup

setup(name='nortok',
      version='0.1.2',
      description='Tokenization and parsing',
      url='http://github.com/Froskekongen/nortok',
      author='Erlend Aune',
      author_email='erlend.aune.1983@gmail.com',
      license='MIT',
      packages=['nortok'],
      package_dir={'nortok': 'nortok'},
      package_data={'mypkg': ['nortok/data/*.*']},
      install_requires=[
          'lxml',
          'nltk',
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],)
