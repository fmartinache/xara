from setuptools import setup

setup(name='xara',
      version='0.1',
      description='Package for eXtreme Angular Resolution Astronomy',
      url='http://github.com/fmartinache/xara',
      author='Frantz Martinache',
      author_email='frantz.martinache@oca.eu',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Professional Astronomers',
          'Topic :: High Angular Resolution Astronomy :: Interferometry',
          'Programming Language :: Python :: 2.7'
      ],
      packages=['xara'],
      install_requires=[
          'numpy', 'scipy', 'matplotlib'
      ],
      scripts=["bin/ker_model_builder"],
      data_files = [('config', ['cfg/discretizor.ui'])],
      include_package_data=True,
      zip_safe=False)

