---
layout: post
title: My Approach in Integrating Excel and Python scripts
category: blog
tags: [excel]
---

## Introduction

In this post I will share how I integrate Python scripts with Excel.

## The Problem

Let's get one thing out of the way: Microsoft Excel is a very powerful tool, because:

- It's very flexible
- It provides instant feedback
- It's pretty fast (unless you write your formulas inefficiently)

I found that Excel formulas already satisfy >90% of my needs. However, on
certain cases it's not enough, and this is where Python is needed. Cases such
as:

- Scraping data from the internet
- Pre-processing some messy data (such as poorly formatted CSV file generated
  from a third-party program)
- Running complex calculations on a huge amount of data

I want to be able to run a macro from Excel (e.g. by clicking a button) and
have the processed data somehow be pasted to the Excel sheet. However, by
default Excel and Python are not connected with each other.

## The Solution

After working several projects that have this problem, I found out that the
ideal solution need to separate as much as possible between the Python script
and the Excel file. The reasons I want to separate/decouple between Excel and
Python are so:

- I can debug/run the Python script without opening the Excel
- I become more confident when editing the Excel file, and not afraid I would break
  the macro (to some extent)

There is an excellent Python library called
[xlwings](https://www.xlwings.org/), but I do not use it since the script would
read or modify the Excel file directly.

The following is my solution to this problem:

1. Have the Python script copy the desired data to clipboard, and have the VBA
   paste from clipboard to a certain location in the Excel file
1. If some "input data" is required for the Python script, then it need to be
   in the form of "command line arguments" (if possible)

## Example Cases

Let's work through a very simple example. In a folder, the files are as follows:

```
+- Folder/
   +-- Document.xlsm
   +-- load_data.py
```

The content of `load_data.py` is:

```python
import pandas as pd
df = pd.DataFrame({'Name':['Joe', 'Mike'], 'Age':[32, 34]})
df.to_clipboard(index=False)
```

Meanwhile, the Excel file has a macro like this:

```vb
Sub LoadData()
    Dim cmd As String
    Dim sht As Worksheet
    ChDrive ThisWorkbook.Path
    ChDir ThisWorkbook.Path

    sht = Sheets("Sheet1")
    cmd = "python load_data.py"

    Dim shell As Object
    Dim errorCode As Integer
    Set shell = CreateObject("WScript.Shell")
    errorCode = shell.Run(cmd, vbNormalFocus, waitOnReturn:=True)
    If errorCode = 0 Then
        sht.range("A1").pastespecial
    Else
        MsgBox "Failed to load data!"
    End If
End Sub
```

Now, if the above macro is run, then the following data will be pasted into Excel:

XXX

## Disadvantage to This Approach

The disadvantage of this approach is that you cannot paste multiple data; so,
you must create a new script. This could be a problem from performance
perspective: invoking a Python script takes a significant amount of time, more
so if you load a large package like Pandas.

## Conclusion

XXX
