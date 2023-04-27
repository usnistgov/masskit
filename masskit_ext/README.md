# masskit_ext

This directory contains masskit functions and utilities which are written in C++ for greater speed and efficiency.

For ease of development, we are using the vcpkg infrastructure for cross-platform convenience.
We are using a submodule, but don't plan to modify vcpkg's git tree, so a lot of the submodule pain should be avoided.
Subtrees would copy too much code into our tree, and it would not be useful for other MSDC users.
The following sections contain some hints for working with this addition.

## Git

### Upon initial clone

This will ensure the code in the linked submodule is also pulled.

```
$ git clone --recurse-submodules git@gitlab.nist.gov:msdc/msdc_services.git
```

### Sync with linked version:

To update an existing repository after pulling.

```
$ git submodule init
$ git submodule update
```
or the following (more succinct?) command:

```
$ git submodule update --init --recursive
```

We don't actually need the `--recursive` argument,
because we don't have submodules of submodules, yet.

### Update the linked version:

```
$ cd libraries/src/masskit_ext/vcpkg/
$ git status
HEAD detached at 2ac61f87f
nothing to commit, working tree clean
$ git checkout master
$ git pull
$ cd ..
$ git status # submodule is now modified, assuming the git pull changed something.)
$ git add vcpkg
$ git commit (vcpkg now points to latest version of master)
```

# Working with vcpkg

## Linux package installation

Ensure flex and bison are installed in the Linux distro
(e.g. sudo apt install flex bison), as well as other standard
development tooling

```
$ ./vcpkg/bootstrap-vcpkg.sh
$ ./vcpkg integrate install
```

### Shared build

Need to create a shared build configuration (aka triplet)
irst, because, reasons.

```
$ cd vcpkg/triplets
$ cp x64-linux.cmake community/x64-linux-dynamic.cmake
$ sed -i 's/static/dynamic/g' x64-linux-dynamic.cmake
$ cd ..
$ ./vcpkg install --triplet x64-linux-dynamic

# We are now using manifest mode, so don't use the following commands
$ ./vcpkg install arrow:x64-linux-dynamic
$ ./vcpkg install pybind11:x64-linux-dynamic
```

To enable access to the shared libraries, execute the following in your bash shell:

```
$ export VCPKG_PATH=$(<$HOME/.vcpkg/vcpkg.path.txt)
$ export LD_LIBRARY_PATH=${VCPKG_PATH}/installed/x64-linux-dynamic/lib:$LD_LIBRARY_PATH
```

### Static build

The static build chain is the default for Linux.
Why? I don't know, there are so many issues with linking
that I still have not solved. So don't actually use this version.

```
 $ cd vcpkg/
 $ ./vcpkg install arrow
```



## Windows package installation

Prefered shared builds, but defaults to an x86 build,
which is not supported by arrow, so need to supply the triplet explicitly.

```
> .\vcpkg\bootstrap-vcpkg.bat
> cd vcpkg
> .\vcpkg integrate install
> .\vcpkg install arrow:x64-windows
> ./vcpkg install pybind11:x64-windows
```

## Notes

1. Linux and Windows binaries can exist in the same tree,
so no worries for using the same source directory tree on Windows/WSL
1. After an update, you must run `bootstrap-vcpkg` again.
Then use the `vcpkg update` command, and follow instructions.
