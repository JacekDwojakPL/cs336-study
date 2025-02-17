---
layout: post
title:  "Initial Commit"
date:   2025-02-15 19:17:27 +0100
categories: general
author: jacek
---
Hello! Here I will post some notes to document my progress (if any) during self-study of CS336: Language Modeling from Scratch. While these noted are shared in the form of this blog,
please be aware that I am not taking any responsibility if you use anything you find here. This space is for me mostly to keep my thoughts organized and document the process of learning. If you find anything useful here - great!

#### Write about learning
The idea of document learning process this I took from Jeremy Howard and book [Practical Deep Learning For Coders](https://course.fast.ai/) where he recommends to do something like that. Since this is some personal space my sincere recommendation is not to use and be inspired by anything you will read here. GitHub Pages allows to host blog from the repository so I decided that I will use Jekyll since it provides nice framework with uncomplicated pipeline and has default GH integration. Of course from the start I stumbeld upon problems with package managers, versioning, persimssions etc. So first advice you can take from here: if you find errors regarding write permissions during run of `bundle install`, eg. something like: 

```shell
Bundler::PermissionError: There was an error while trying to write to
`/usr/local/lib/ruby/gems/3.4.0/cache/public_suffix-6.0.1.gem`. It is likely that you need to grant write permissions for that
path.
``` 
make your bundle configuration to install dependencies to local directory with `bundle config --local path .bundle` and then run `bundle install` to install dependencies locally. It solved my problem and allowed me to preview this site on localhost. Just remember to add `.bundle` directory to `.gitignore`. 

#### CS336 Language Modeling from Scratch
It is Stanford University course which takes it's students through process of creating Large Language Model from scratch. Lectures and code assignments are publicly shared [here](https://stanford-cs336.github.io/spring2024/). I am not sure if the lectures on the campus are recorded but from what can be found on the website lectures have the form of executable code which student follows with the debugger. Quite unusual but after I finished the first lecture I can say that it's pretty good form of study. It allows you to follow the code directly with all variables and application state available to examine. 

First lecture is an introduction to the course with some history context - it builds general landscape of the subject and introduces vocabulary and terms. Next the general framework of creating LLMs is outlined. From gathering raw text data through preparation of it to the form digestable by the computer. Large part of the lecture is devoted to explaination of how Byte Pair Encoding works - building the BPE Tokenizer is part of first assignment so the algorithm and simple example are presented to develop some intuitions. Also different approaches to the tokenizeing the text are discussed along with answer to the question of why tokenization is needed in the first place. Lecture concludes with some administrivia for on-campus students. I finished the first part of the first assignment - building the BPE Tokenizer which I will share next time.