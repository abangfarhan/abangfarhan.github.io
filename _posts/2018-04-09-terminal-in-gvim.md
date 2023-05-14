---
layout    : post
title     : Spawning Terminal inside gVim
subtitle  : Why nobody told me about this before?
date      : 2018-04-09 17:15
published : true
author    : abangfarhan
category  : blog
tags      : [programming, vim, text editor]
---

Apparently you can create a terminal when you are using gVim (GUI version of Vim, which is the version that I used on Windows). All you have to do is execute the command `:terminal`, and a new split will be created to create the terminal. On Windows, the default terminal that would be created is the Windows command prompt. But, if you have installed bash you can spawn bash too by typing `:terminal bash`. Or, if you want to create a Python interactive shell, you just have to type `:terminal python`. Below is the screenshot:

![screenshot]({{site.baseurl}}/img/2018-04-09-terminal-in-gvim/00.gif)


You can still move between windows and enter `ex` mode. For more information, just type `:help terminal`.
