README
---

The detailed description of datafile 'data.TD_RvNN.vol_5000.txt' is as follows.

The datafile is in a tab-separated column format, where each row corresponds to a tweet.
Consecutive columns correspond to the following pieces of information.

```
<Excerpt>

656955120626880512	None	1	2	9	1:1 3:1 164:1 5:1 2282:1 11:1 431:1 473:1 729:1
656955120626880512	1	2	2	9	0:2
624298742162845696	None	3	72	26	1:1 34:1 3:1 71:1 9:1 202:1 11:1 140:1 12:1 624:1 124:1 1266:1 692:1 670:1 90:2 3932:1 1246:1
624298742162845696	3	41	72	26	0:1 33:1 8:1
624298742162845696	3	66	72	26	0:1 186:1 203:1 6:1 46:1
```

1. root_id: an unique identifier describing the tree (tweet_id of the root)
2. index_of_parent_tweet: the index number of the parent tweet for the current tweet
3. index_of_the_current_tweet: the index number the current tweet
4. parent_number: the total number of the parent node in the tree that the current tweet is belong to
5. text_length: the maximum length of all the texts from the tree that the current tweet is belong to
6. list_of_index_and_counts: the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Twitter)
