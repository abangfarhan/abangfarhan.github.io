---
layout    : post
title     : Making Scripts of Any Language a Global Executable on Windows
subtitle  : Little trick that made my life easier.
date      : 2018-01-28 07:38
author    : abangfarhan
category  : blog
tags      : [programming, windows, batch]
---

On a \*nix environment, if you have a script created using any programming language, and you want to make that script an executable on Windows, you only need to put a shebang line on top of the script. For example, on a Python script you would need to put something like this:

```text
#!/usr/bin/env python3
```

Basically, it tells the computer how to run this script. Furthermore, if you want that script to be executable (executable from the command line, that is), you only need to put that file in the [PATH environment variable](https://superuser.com/questions/284342/what-are-path-and-other-environment-variables-and-how-can-i-set-or-use-them) (such as `/usr/bin`).

Unfortunately, if you are on Windows the shebang line will be ignored. As a result, if you want to execute a Python script from the command line, you have to execute `python my_script.py`. Luckily, there is a workaround to this. You only need to create a batch file that will execute the script, and put that batch file in a folder that is included in the PATH environment variable.

Suppose you really want to use [Ack](https://beyondgrep.com/), a very fast and elegant program to search trough files (like grep). The program has a single executable, written in Perl. The installation instruction for Windows tells us to install Ack using Chocolatey package manager. But for some reasons you haven't and don't want to use Chocolatey. So you need to find a way to make Ack a global executable.

First, you need to create a folder to put all of your scripts. For example, the folder is on `D:\scripts`. Next you need to include this folder in the PATH environment variable. (There are many instructions to do this on the Internet.) Suppose you put the Ack script on that folder. To make this script an executable, you need to create a batch file that contains this:

```
@echo off
perl "D:\scripts\perl.pl" %*
```

I think the above script is pretty much self explanatory. Except, perhaps, the `%*` part. What it does is parsing in all arguments you give to the program. Oh, and you have to save the file as a `.bat` file. You probably should call name the batch file the same as the original script. In this case it should be `ack.bat`.

As you can see, we use `perl` on the script because `Ack.pl` is written in Perl, and `perl` is the program to run Perl scripts. If the script is written in Python, the batch file above will be written as `python D:\scripts\my_script.py`.
