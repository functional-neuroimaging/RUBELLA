# Contribute new code using Git

## Getting started with basic git commands
```sh
# create your own local version of the repository
$ git clone https://github.com/functional-neuroimaging/RUBELLA.git
$ cd RUBELLA # move into your copy of the git repository
$ git status # tells you what branch you are on, and currently tracked/untracked files
  On branch main
  Untracked files:
    (use "git add <file>..." to include in what will be committed)
          contributing.md
          dependencies.txt
$ git pull origin main # update your local main branch to match the online repository, i.e. if new code was committed online
```

## Steps to follow when committing new code
```sh
# VERY IMPORTANT - create a new branch for your own dev, this will prevent introducing potential bugs on the main branch, where everybody else is working
$ git branch my_branch_for_XYZ
# change your git branch - you will now only modify this new branch, and not the main branch
$ git checkout my_branch_for_XYZ
###NOW APPLY YOUR CHANGES###
$ git status # check what you have changed so far
  On branch my_branch_for_XYZ
  Untracked files:
    (use "git add <file>..." to include in what will be committed)
          my_new_function.py

$ git add my_new_function.py # add new files/folders to the package, on your current branch

$ git status # now it is tracked
  On branch my_branch_for_XYZ
  Changes to be committed:
    (use "git restore --staged <file>..." to unstage)
          new file:   my_new_function.py
# commit your changes
$ git commit -m "the new script my_new_function.py does XYZ"
$ git status # all tracked changes are commited
  On branch python_package
  nothing to commit, working tree clean
# push your commit(s) to the online repository
$ git push origin my_branch_for_XYZ
# you are done for local steps, the rest happens on GitHub, where we can merge your branch with the main branch
```

