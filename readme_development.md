# Local build

The old method, `python setup.py develop` is obsolete, use:

```pip install -e .```

instead. When done, 

```pip uninstall masskit```

If you are working in the python code, you only need to do this once. If modifying the C++, you'll need to do it everytime to recompile.

## Debugging the build

To see all of the messages from the build process, use the following command:

```VERBOSE=1 pip install --no-cache-dir -v -e .```

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
