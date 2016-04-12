---
layout: post
comments: true
title:  "Faster grep"
date:   2016-04-11 10:10:00
disqus_identifier: 20160111
categories: ['Linux', 'GNU grep']
---

Grep is GNU tool that is used for matching lines with particular pattern.
In most cases it is remarkably fast but sometimes it is better to use explicit implementation mimicking grep's functionality.

If one wants to search for not only one pattern, grep's option *-f* allows to specify file containing arbitrary number of patterns, one per line.
Following command searches for patterns from *patterns.txt* in *file.txt*.
Lines that are matched with any pattern are printed into standard output.

```bash
grep -f patterns.txt file.txt
```

GNU grep's approach of matching is general but sometimes at the expense of speed.
Uderstanding of data and details of file we work with can speed up our pattern matching.

## Example
Let's say that we have a file with paths to images but we want to obtain just some of them.

*file.txt*

```
/images_aug/2008_000002.jpg /labels_bbox/2008_000002.png
/images_aug/2008_000003.jpg /labels_bbox/2008_000003.png
/images_aug/2008_000007.jpg /labels_bbox/2008_000007.png
...
```

We also have a file with patterns (file names of images) that we want to include in result.

*patterns.txt*

```
2007_000032.png
2007_000039.png
2007_000063.png
...
```

In both files we assume that every line is unique and lines are sorted.
Without uniqueness this particualar example wouldn't make sense.
Sorting lines is one of the secrets allowing much faster pattern matching, therefore sorted lines are necessary.
If we use GNU grep lines don't have to be sorted but sorting itself doesn't affect overall speed of pattern matching because we can sort files just once.
Sorting (GNU sort) file  with 11,355 lines takes about 0.080 seconds and that is insignificant in comparison with the time spent for pattern matching. 

<script src="https://gist.github.com/martinkersner/32e909a518d6983d37b524a4d3984dd4.js"></script>

## Evaluation
In order to compare GNU grep and faster grep I used different number (100; 500; 2,500; 5,000 and 10,000) of patterns.
Patterns where searched in file with 11,355 lines.
Plot below depicts times spent for pattern matching of both methods.

<img src="http://i.imgur.com/qtW8mcR.png?1" align="center"/>

GNU grep seems to perform faster when number of patterns is lower but with higher number of patterns time spent with matching dramatically increases.
Faster grep is significantly faster when number of patterns is rather high.
