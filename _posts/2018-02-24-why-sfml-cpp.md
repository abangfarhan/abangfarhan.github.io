---
layout: post
title: Why you should use the SFML library to learn C++
subtitle: Something I learned in the past couple of months.
category: blog
tags: [programming, cpp, SFML]
---

Hi guys, in this post I am going to talk about SFML, or Simple and Fast Multimedia Library, and how it helped me learn more about the C++ programming language. But, first of all, what is SFML?

## What is SFML?

[SFML](https://www.sfml-dev.org/) is a library that you can use to make graphics. I think SFML can be categorized as a high-level library, since the API (i.e. the functions, classes, etc.) of the library is very simple yet capable of doing things that usually require you to write many lines of codes (especially in C++). Although I use SFML with C++, actually it has other bindings to other languages, such as C, Java, Python, and more. 

Did I said that it was a graphics library? Well, that's not entirely true, because it is much more than that. The official website said that SFML is "a simple interface to the various components of your PC, to ease the development of games and multimedia applications." Accordingly, it has five modules that can easily speed up the development process: system, window, graphics, audio, and network. But for now I only primarily use the graphics library.

## How do I get it?

To get the SFML, you can get the latest stable version from [here](https://www.sfml-dev.org/download/sfml/2.4.2/). For now you should find out by yourself how to get it working, e.g. by googling or reading the docs. The download page provides many versions depending on your compiler, such as Visual C++11, GCC, and so on. In the future I will write a post to help you make the library from source, since it is very easy and very important to be learned by beginners. For now, you can check out this [Youtube video](https://www.youtube.com/watch?v=UM93glM0Fhs) (this is how I first learned about building library from source and including it in my program).

## Why should I use it?

There are several reasons why you should use this library, especially if you are a beginner in C++. First, it is easy to use. Second, you can use this library to visualize stuffs. Third, you may learn important things that are important to learn C++.

As I said, the API of SFML is easy to understand and easy to use. For example, to create a simple shape, like a rectangle, you only need to write

```cpp
sf::RectangleShape rectangle;
rectangle.setSize(sf::Vector2f(100, 50));
rectangle.setOutlineColor(sf::Color::Red);
rectangle.setOutlineThickness(5);
rectangle.setPosition(10, 20);
window.draw(rectangle);
```

I know, I know, it doesn't look very simple. But this is C++, and if you compare SFML with other graphics library which is usually more low-level, you will understand that SMFL is relatively simple.

Another reason why SFML is easy to use is because the documentation is easy to navigate. Seriously. You can easily find what you need very quickly. The [documentation](https://www.sfml-dev.org/documentation/2.4.2/modules.php) is divided for each module, and in each module's documentation you can find all the classes' names. Moreover, each class name is pretty much self explanatory, like `sf::Font`, `sf::CircleShape`, and so on. Each documentation of each class also shows a detailed description, which usually shows an example. If you just use Google to create something with SFML, you are very likely to immediately find the answer too.

The second reason why you should use SFML is that you can use this library to visualize stuffs. What I meant is that you can implement algorithms or various programming problems and also visualize it. Some algorithms, like [shortest path algorithms](https://en.wikipedia.org/wiki/Shortest_path_problem#Algorithms), or [maze generation algorithms](https://en.wikipedia.org/wiki/Maze_generation_algorithm), are much better if visualized. You can literally see your implementation of an algorithm in action! If you have used p5.js before to do this kind of things, you will surely enjoy using SFML too.

The third reason to use SFML is that you may learn other important things. For me, SFML allow me to learn better about:
1. **Separating components of a program**: in a small C++ program, a script in a single file is often enough. But as your program grows in complexity, it is generally a good practice to separate each component in separate files.
2. **Using Makefile**: because the program is separated on various scripts, it will be tedious and fragile if I compile the program manually from the command prompt. Luckily, I can use the [make program](https://www.gnu.org/software/make/). If you install GCC using MinGW or TDM-GCC, the make program is called `mingw32-make`. Previously, I thought Makefile was really hard to use. It turns out that it is pretty simple and easy to use (at least for my current purposes, which is also very simple).
3. **Debugging program using GDB**: [GDB](https://www.gnu.org/software/gdb/) is usually come preinstalled when you installed MinGW GCC. Previously, when my program went wrong I will sprinkle `std::cout` around to find the problem. Needless to say, that is not the most efficient way to debug any program. With GDB, you can run your program and stop it anytime you want. You can also inspect any variable that you have created in your program. SFML supports debugging, so it is possible to run your graphics program, which requires an infinite loop to display the window continuously, while still capable to debug your program step by step using GDB.

## What have I created using SFML?

As of the time of this writing I have created three Github repositories that used SFML library:

1. [Game of life](https://github.com/abangfarhan/game-of-life-sfml): in this repo I implemented the Conway's game of life.
2. [Graphs](https://github.com/abangfarhan/graph-sfml): in this repo I implemented various algorithms related to graphs. Firstly, I created a data structure that I think will suffice for my purpose. Then, using that data structure I learned how to display each node to display the graph. When the displaying was done, I implemented some algorithms so that it can be visualized. The algorithms that I have created so far was Djikstra's shortest path algorithm and Prim's spanning tree algorithm.
3. [Maze](https://github.com/abangfarhan/maze-sfml): in this repo I created maze using SFML and also implemented some algorithms. Just like when I created graphs, I also need to design the data structure first, then I need to properly display the maze using SMFL. Finally, I can implemented some algorithms related to mazes. So far I have implemented the bactracking and Kruskall algorithm to generate a maze, and the wall following algorithm to solve a maze.
