+++
title =  "git cheatsheet"
tags = ["git", "kunal_kushwaha"]
date = "2020-11-28"

+++

started learning git basics started with some git basics

> cd 

for changing directory

> ls 

for listing out directory

> mkdir

for making new directory

> git init

for intialising empty git repo

> ls -a

to see hidden files

> touch file_name

to create a new file in directory

> git status

for knowing changes made in repository

> rm a b
> rm *.txt    

to remove pattern of files in repo

> rm a/
> rm -R a/

to remove directory in repo

> rm -i a.txt

to remove by confirmation

> rm -rf B

When you combine the -r and -f flags, it means that you recursively and forcibly remove a directory (and its contents) without prompting for confirmation.


> git add .

here all files get added that are modiefied  to maintain history of files

> git commit -m "message about commit"

here we commit all files

> git restore --staged file_name

here it is all files are removed whuch are added in git add command

> git log 

here it tells about all last commits history 

> git reset  ssh_id

deletes all commits above the paricular id

> git stash 

here for the files which cant be lose or cant be commited we just hide them

> git stash pop

uncomited files are called and 

> git remote add origin link_of_the_repo

adding the remote fiiles to origin

> git remote -v 

links related to this repo

> git branch feauture

new branch gets modified 

> git checkout feature

changes head to new branch commits are made on that

> git remote add upstream link_repo

url which we have forked 

for new pull reques and new bug create new branch and it can go on it is rule create new branch

> git pull upstream main

to fetch all details from main account or forked account


