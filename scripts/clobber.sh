#!/usr/bin/env bash

# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This script removes all files/directories that are not tracked by git
# except the following files/directories
EXCLUDE_LIST='.vscode/ .env'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
NEML2_DIR=$(dirname "$SCRIPT_DIR")

# go to the git root
cd $NEML2_DIR

# files/directories to clobber
FILES_ALL=$(git ls-files . --ignored --exclude-standard --others --directory)

# filter out excluded files/directories
FILES=()
COUNT=0
for FILE in $FILES_ALL; do
  if [[ ! " ${EXCLUDE_LIST[*]} " =~ " $FILE " ]]; then
    FILES+=($FILE)
    COUNT=$((COUNT+1))
  fi
done

# check if there are any files to remove
if [ $COUNT -eq 0 ]; then
  echo "No files to remove."
  exit
fi

# remove files
echo "The following directories/files are going to be removed:"
echo "--------------------------------------------------------"
printf "%s\n" "${FILES[@]}"
echo "--------------------------------------------------------"

# prompt user
while true; do
    read -p "Do you wish to remove the above directories/files? [y/n] " yn
    case $yn in
        y ) rm -rf ${FILES[@]}; break;;
        n ) exit;;
        * ) echo "Please enter y (for yes) or n (for no).";;
    esac
done
