""" Package installer """
import fnmatch
import os
import sys
from setuptools import setup, Extension

# Requiring python 3.4+.
# To simplify code for Tornado coroutines return statements, we don't support Python 3.3
# ( more info here: http://www.tornadoweb.org/en/stable/guide/coroutines.html#coroutines ).
if (sys.version_info.major, sys.version_info.minor) <= (3, 3):
    print("This Python is only compatible with Python 3.4+, but you are running "
          "Python {}.{}. The installation will likely fail.".format(sys.version_info.major, sys.version_info.minor))

# ------------------------------------
# Configuration
PACKAGE_NAME = 'diplomacy-research'
PACKAGE_VERSION = '1.0.0'

def has_libprotobuf_dev():
    """ Detects if the libprotobuf-dev package is installed """
    c_path = os.environ.get('CPATH', '')
    paths = c_path.split(':') + ['/usr/include', '/usr/local/include']
    for path in paths:
        if os.path.exists(os.path.join(path, 'google/protobuf/stubs/common.h')):
            return True
    return False

def find_proto_cc_files():
    """ Finds the c++ files in the proto folder """
    # Source: https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python/2186565#2186565
    matches = []
    for root, dirnames, filenames in os.walk('diplomacy_research/proto'):
        for filename in fnmatch.filter(filenames, '*.cc'):
            matches.append(os.path.join(root, filename))
    return matches

def find_tf_user_ops():
    """ Find the TF c++ user ops files """
    # Source: https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python/2186565#2186565
    matches = []
    for root, dirnames, filenames in os.walk('diplomacy_research/models/user_ops'):
        for filename in fnmatch.filter(filenames, '*.cc'):
            matches.append(os.path.join(root, filename))
    return matches

def get_module_exts():
    """ Returns the module extensions to install """
    import tensorflow as tf
    extensions = []
    root_dir = os.path.dirname(os.path.realpath(__file__))
    user_ops_dir = os.path.join(root_dir, 'diplomacy_research', 'models', 'user_ops')

    # Protobuf C++ extensions
    if has_libprotobuf_dev():
        extensions += [Extension('cpp_proto',
                                 sources=['diplomacy_research/proto/cpp_proto.c'] + find_proto_cc_files(),
                                 language='c++',
                                 include_dirs=[root_dir, os.path.join(root_dir, 'diplomacy_research', 'proto')],
                                 extra_compile_args=['-std=c++11'],
                                 extra_link_args=['-std=c++11'],
                                 libraries=['protobuf'])]

    # TensorFlow C++ User Ops
    for user_op_path in find_tf_user_ops():
        user_op_name = user_op_path.split('/')[-1].split('.')[0]
        extensions += [Extension(user_op_name,
                                 sources=[user_op_path],
                                 language='c++',
                                 include_dirs=[user_ops_dir, tf.sysconfig.get_include(), tf.sysconfig.get_lib()],
                                 extra_compile_args=['-std=c++11'] + tf.sysconfig.get_compile_flags(),
                                 extra_link_args=['-std=c++11'] + tf.sysconfig.get_link_flags())]
    return extensions

setup(name=PACKAGE_NAME,
      version=PACKAGE_VERSION,
      author='Philip Paquette',
      author_email='pcpaquette@gmail.com',
      packages=[PACKAGE_NAME.replace('-', '_')],
      install_requires=[
          'bs4',
          'diplomacy==1.1.0',
          'grpcio==1.15.0',
          'grpcio-tools==1.15.0',
          'gym>=0.9.6',
          'h5py>=2.8.0,<2.10.0',
          'html5lib',
          'hiredis',
          'numpy>=1.15,<1.16',
          'protobuf==3.18.3',
          'pymysql',
          'python-hostlist',
          'pytz',
          'pyyaml',
          'redis',
          'regex',
          'requests',
          'tabulate',
          'tensorflow==1.13.1',
          'tensorflow-probability==0.6.0-rc1',
          'toposort',
          'tornado>=5.0',
          'tqdm',
          'ujson'
      ],
      ext_modules=get_module_exts())

# ------------------------------------
