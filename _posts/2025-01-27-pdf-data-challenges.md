---
layout: post
title: Challenges When Reading Tabular Data in PDFs
category: blog
tags: [programming]
---

## The Challenges

Correctly reading tabular data from PDFs is _extremely challenging_ for the following reasons:

- Different table layouts in each PDF means you must handle all possible formats.
- Table layout may change in the future, causing the program you have created to not work.
- Even if the formats are consistent, these issues may cause inaccurate parsing:
    - Multi-line cells (i.e. when a cell in the table contain multiple lines, either because the text is too long, or the cells are deliberately merged horizontally or vertically).
    - Invisible borders: if the tables have no visible borders, the program may have problems distinguishing between the cells.
    - Cells that are too close to each other may cause the program to think that they belong to a single cell.
    - Invisible texts: sometimes a table has invisible text (for example, the text has white font color, and the background is also white). We generally don’t want to read these texts; however, the parser program usually still read them.
    - The PDFs may contain multiple tables, so you must correctly identify which table to parse.
    - A table may span multiple pages in the PDF.
    - The text inside a cell may “overflow” to next cell, which cause the text to either be hidden or mixed with the next cell’s text.

Unlike other data formats such as CSV or JSON, which store data in an explicitly structured and machine-readable manner, PDFs prioritize layout and visual representation, making data extraction non-trivial.

## Some Libraries

The libraries below can be used for PDFs that still have copy-able texts in them. If the texts inside the PDF are not copy-able (e.g. from scanned documents), then you must use OCR (optical character recognition) programs (which is an even more challenging task, since OCR itself may be inaccurate at reading the characters). Note that you should avoid using OCR if the PDFs still have copy-able texts. 

- **Open-source (free) solutions**:
    - Tabula: <https://github.com/tabulapdf/tabula-java>
        - The library is written in Java, but there's a python binding (<https://pypi.org/project/tabula-py/>).
        - Pros: very fast even when handling very large tables.
        - Cons: require Java installation; I have also found that it sometimes cannot parse multi-line cells.
    - Camelot: <https://camelot-py.readthedocs.io>
        - Cons: extremely slow when parsing large tables (like 1 hour-slow for a page containing hundreds of cells). This is the reason that I don't use this library. If your tables are not large, then you can consider to use this library.
    - Pdfplumber: <https://github.com/jsvine/pdfplumber>
        - I used this library for parsing tables from [IDX](https://idx.co.id/)’s 5% ownership report (which is large, containing invisible borders, merged cells, etc.).
        - This library can extract the correct text given the X & Y coordinates and the size of the bounding box. For example, you can just say that you want to read the text inside a box area starting at coordinates (50, 100) with 50 points width and 40 points height.
        - This library can also find table cells (see [the documentation here](https://github.com/jsvine/pdfplumber?tab=readme-ov-file#table-extraction-methods))
        - This library is pretty slow and requires lots of coding to correctly read in the data.
- **Paid solutions** (note that I haven’t tried any of these programs):
    - PDF Tables: <https://pdftables.com/>
        - For pricing see <https://pdftables.com/pricing> ($50 per 1000 pages for the lowest tier, which amounts to ~Rp800 per page)
        - Has APIs that can be used from Python or other programming languages. 
        - Apparently, this has no free demo that you can try.
    - Google’s Document AI: <https://cloud.google.com/document-ai/>
        - For demo, go to <https://cloud.google.com/document-ai/docs/try-docai>
        - This also has APIs that can be used from Python etc.

