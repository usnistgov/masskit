#!/bin/bash
set -eux
shopt -s expand_aliases

SERVERIP=$1

alias new_ssh='ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no msdc@$SERVERIP'
alias new_rsync='rsync -e "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"'

new_ssh mkdir -p nist_service
new_rsync -avP --exclude __pycache__ \
          --exclude anaconda3 \
          --exclude build \
          --exclude inst_info.json \
          * msdc@$SERVERIP:nist_service
new_ssh "cd nist_service && sudo scripts/setup_nist_service.sh"
