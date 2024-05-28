# An empirical study of the misunderstanding of IP laws on social media
This research is supported by NCSA Illinois Computes program, IC-382

## background

## Automatic annotation
During this process, we will use machine-learning approach to process texts from social media, and identify the possible misunderstandings in the text. We will use the manual annotations as input.
- expert annotation
- expert + student annotation

We will first train the model with expert annotation (high accuracy, small sample size), and then, use a heuristic approach to evaluate a random subset of student annotation. With expert judge of machine-prediction vs student annotation, we can obtain a better student annotation set to expand our training set. Then, we will train again with the expanded training set, and work with student annotation again. Eventually, we can train a good annotation set.


