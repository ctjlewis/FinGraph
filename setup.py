import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='FinGraph',  
     version='0.0.2',
    #  scripts=['dokr'] ,
     author="C. Lewis",
     author_email="ctj.lewis@icloud.com",
     description="A simple abstraction layer on top of Matplotlib that allows for easily displaying financial graphics.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/ctjlewis/FinGraph",
     py_modules=['FinGraph'],
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )