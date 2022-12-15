# Local build

The old method, `python setup.py develop` is obsolete, use:

```pip install -e .```

instead. When done, 

```pip uninstall masskit```

If you are working in the python code, you only need to do this once. If modifying the C++, you'll need to do it everytime to recompile.


# Installing C++ Arrow

## OS

See https://arrow.apache.org/install/

## Python

```pip install pyarrow==10.*```

or

```conda install arrow-cpp=10.* pyarrow=10.* -c conda-forge```


# Building C++ standalone.

In masskit_ext:

```
cmake -S . -B build/debug -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/debug

  or 

cd build/debug
ninja
```
## Options

Clang:
```
CC=clang CXX=clang++ cmake ..
```

Debugging the build:
```
cmake --build . --verbose
```

Debugging the config:
```
cmake --trace ..
cmake --trace-source="CMakeLists" ..
```


### Misc

- -DCMAKE_BUILD_TYPE= Pick from Release, RelWithDebInfo, Debug, or sometimes more.
- -DCMAKE_INSTALL_PREFIX= The location to install to. System install on UNIX would often be /usr/local (the default), user directories are often ~/.local, or you can pick a folder.
- -DBUILD_SHARED_LIBS= You can set this ON or OFF to control the default for shared libraries (the author can pick one vs. the other explicitly instead of using the default, though)
- -DBUILD_TESTING= This is a common name for enabling tests, not all packages use it, though, sometimes with good reason.


# Git

## List all remotes
```
git remote -v
```

## Syntax to add a git remote
```
git remote add REMOTE-ID REMOTE-URL
```
e.g.
```
git remote add github git@github.com:usnistgov/masskit.git
git branch -M main
git push -u github main
```