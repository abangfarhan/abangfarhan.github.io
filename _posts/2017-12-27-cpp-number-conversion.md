---
layout    : post
title     : Weird C++ Number Conversion
subtitle  : At least it's weird for me...
date      : 2017-12-27 08:55
published : true
author    : abangfarhan
category  : others
tags      : [cpp, C++, programming]
---

While learning the C++ programming language, I found out that there are three ways to convert number types. For instance, suppose you want to convert an `int` to a `float`, then you can do either of the followings:

```cpp
int myInt = 10;
float myFloat1 = (float) myInt;
float myFloat2 = float(myInt);
float myFloat3 = static_cast<float>(myInt);
```

As explained everywhere on the Internet, the first method is a C-style method, while the second and the third are C++ methods. But, are they completely interchangeable? Try this:

```cpp
float myFloat1 = (float) 1/2;
float myFloat2 = float(1/2);
float myFloat3 = static_cast<float>(1/2);
```

That is, we are doing a division of two `int`s, and we want the result to be a `float`. However, try to print each variables that we just created!

```cpp
std::cout << myFloat1 << std::endl; // prints 0.5

std::cout << myFloat2 << std::endl; // prints 0

std::cout << myFloat3 << std::endl; // prints 0
```

Now, why do `myFloat2` and `myFloat3` equal to zero? I think this is due to how the difference in the mechanism of the conversion that we just did. This is my explanation of what happens:

1. When we do `(float) 1/2`, the compiler will first convert each number to a float, and then execute the division. In this case, `1` is converted to `1.0`, `2` is converted to `2.0`, and finally the compiler divides `1.0` with `2.0`. The result is a `float` that equals to `0.5` since both numbers are `float`s.
2. When we do `float(1/2)` or `static_cast<float>(1/2)`, the compiler will first execute the divison, and then convert the result to `float`. In this case, the compiler will divide `1` (an `int`) with `2` (an `int`), and the result will be `0` (since `0.5` rounded down to the nearest integer is `0`). After that, the number `0` is converted to `float`. So, the final result is still `0`.
