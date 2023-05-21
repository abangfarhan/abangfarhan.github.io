---
layout    : post
title     : C/C++ Programming with Curses on Windows
subtitle  : How to develop C/C++ programs with curses, on Windows!
date      : 2018-01-27 07:38
author    : abangfarhan
category  : blog
tags      : [programming, windows, c, cpp]
---

[Curses](https://en.wikipedia.org/wiki/Curses_(programming_library)) is cool. It enables us to make cool command-line applications. Consider this demo program:

<!-- ![tuidemo](..\img\2018-01-27-windows-curses\00.gif) -->
![tuidemo]({{site.baseurl}}/img/2018-01-27-windows-curses/00.gif)

To develop such programs on Windows, we can use PDCurses library.	You can download the latest stable version [here](https://sourceforge.net/projects/pdcurses/files/). Alternatively, you may also clone the Github repository to get the latest nightly version on [this repo](https://github.com/wmcbrine/PDCurses).

Download the source and unpack it. You will see a folder called `win32` in the unpacked files. Go to that directory. There you will see some files needed for making the library. I am going to assume that you use mingw32, like I do. To build the library, you need to open command prompt and go to this directory, and execute:

```
mingw32-make -f mingwin32.mak pdcurses.a
```

After that, you will notice that several files will be generated (such as `addch.o`, `addchstr.o`, etc.). The one you need is `pdcurses.a`. This is the library. To test whether the built library works or not, do the following steps:

1. Create a new directory somewhere else. Let's call it `pdcurs34-test`.
2. Copy the `pdcurses.a` file here.
3. Create a new subdirectory here called `include`. This folder will hold all the necessary header files.
4. Go back to the `pdcurs34` source folder. On the root folder you will see several header files (`curses.h`, `curspriv.h`, `panel.h`, and `term.h`). Copy all of these files to the `include` folder that you just created.
5. Finally, copy all of the `.c` files from `demos` folder to the `pdcurs34-test` folder. This is only needed for testing.

The preparation is complete, now to actually test, open command prompt and go to `pdcurs34-test` directory. We will try to build one program, the one called `worm.c`. To do that, you only need to execute

```
gcc -o worm worm.c pdcurses.a -I include
```

And that's it! If it work successfully, you can see the built program called `worm.exe` and run it from your command prompt. Having done this, now you can learn how to build command line programs using C/C++.

Have a nice day!
