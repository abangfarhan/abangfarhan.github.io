---
layout: post
title: Navigating the Web without a Mouse
category: blog
tags: [others]
---

## Introduction

In this post I just want to share how I managed to use my browser with
only my keyboard (most of the time).

Most of you must have already known about basic web browser's (like
Mozilla Firefox, for instance) key bindings, such as using `up` and
`down` key for scrolling, or using `PageUp` and `PageDown` for
scrolling with greater distance.

Below are some of the common key bindings that I used requently in my
browser.

1. `Alt-D` to focus on the URL bar.
2. `Ctrl-Tab` to go to the next tab.
3. `Ctrl-Shift-Tab` to go to the previous tab.
4. `Ctrl-<num>` to go to tab number `<num>`.
5. `Ctrl-Shift-PageUp` to move the current tab to the left (use `PageDown`
   instead of `PageUp` to move to the right). Sadly the movement doesn't
   loop (for example if your tab is already on the leftmost position,
   and you try to move it to the left, it won't change to the
   rightmost position).
6. `Ctrl-W` to close current tab.
7. `Ctrl-F` to find text.
8. `Ctrl-R` to refresh page.

I also want to mention that I press my `Ctrl` key with my left palm,
so my hands can stay in the same position most of the time.

## Clicking Links and Copying Texts

The real challenge when not using mouse is how to click links and copy
texts. In the past I have tried using browser extension that emulates
Vim keybindings, but got disappointed because suddenly my browser
stopped supporting that extension. Anyway, I actually prefer to use
keybindings that will work on most browsers and not just on my browser
(so I can use other people's computers without having to conciously
forgetting my browser's keybindings). By the way, in that extension to
click a link all you have to do is press `F`, then all visible links
will be overlayed with certain letters, and you have to type the
letters corresponding to the link you want to click. Very convenient.

Luckily, you can sort of emulate that extension's clicking mechanism
by using `Ctrl-F`. Indeed, `Ctrl-F` is the keybinding to find a text. For
example, you want to click a link with the text [like this](https://www.duckduckgo.com). First, you
click `Ctrl-F` and put in the words in that link (e.g. "like this"). If
the search doesn't immediately go to that link, just keep pressing
enter. Most of the times the text in the link is unique, so you just
have to press `Enter` a few times. When the search focus on the link you
want to click, you just have to press `Escape` (to stop the search) then
click `Enter` to open the page in the current tab, or `Ctrl-Enter` to open
the page in a new tab.

The mechanism above can also be used to copy texts. First, search the
words that's located in the starting position of the text (for
example, search "The mechanism" to copy this paragraph). If you're on
the correct position, press `Escape`, otherwise keep pressing
`Enter`. Notice that the words you search will be already highlighted in
the page, now all you have to do is press `Shift-Right`, `Shift-Down`,
etc. until all the desired text is highlighted.

There is one drawback to the clicking link mechanism I used,
though. Some "links" on a page are actually not a link. I found that
some sites have clickable texts that really look like a hyperlink, yet
it's not. Instead of it being inside an `<a>` tag, it's just a regular
`<span>` that has an `onClick` event. In such cases the "links" can only
be clicked using a mouse. *\*Sigh\**.

## Focusing on Input Box

Weirdly, there is no keybinding to focus on the input box in a
page. Since I need this functionality so bad I resort to using a
browser extension called [Fox Input](https://addons.mozilla.org/en-US/firefox/addon/fox-input/) (Firefox), so I just need to type
`Alt-I` to do the task.

## Closing Remarks

I think that's it for this post. I hope you gain some useful information.
