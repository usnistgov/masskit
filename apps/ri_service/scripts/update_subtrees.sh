#!/bin/bash
git subtree pull --prefix=nistms2 https://gitlab.nist.gov/gitlab/msdc/nistms2.git master --squash
git subtree pull --prefix=PA-Graph-Transformer https://github.com/burntcobalt/PA-Graph-Transformer.git master --squash
