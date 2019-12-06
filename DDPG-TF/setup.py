from setuptools import setup

setup(name='DDPG-TF',
      packages = ['ddpg'],
      version = '2.0.3',
      author='Dekki',
      author_email = 'dekkiaero@gmail.com',
      url = 'https://github.com/Dekki-Aero/DDPG',
      description = 'DDPG implimentaion in Tensorflow-2.0 ',
      long_description=open('README.txt').read(),
      long_description_content_type="text/markdown",
      keywords = ['Deep Determnistic policy gradient','Actor Critic','Reinforcement Learning','DDPG'],
      install_requires=['tensorflow==2.0','gym','numpy','matplotlib'])