# Template / object / article effects on failure

N = 300, failures = 61. p-values are permutation chi-square (B = 20000); post hoc and unadjusted across the tests below.

## Template, all prompts
chi2 = 2.23, permutation p = 0.5342

|   template_id |   n |   failures |   fail_rate |
|--------------:|----:|-----------:|------------:|
|             0 |  73 |         13 |       0.178 |
|             1 |  76 |         15 |       0.197 |
|             2 |  65 |         11 |       0.169 |
|             3 |  86 |         22 |       0.256 |

## Template, keys prompts only
chi2 = 5.00, permutation p = 0.2454

|   template_id |   n |   failures |   fail_rate |
|--------------:|----:|-----------:|------------:|
|             0 |  12 |         12 |       1     |
|             1 |  15 |         14 |       0.933 |
|             2 |   9 |          7 |       0.778 |
|             3 |   9 |          9 |       1     |

## Template, non-keys prompts only
chi2 = 15.92, permutation p = 0.0010

|   template_id |   n |   failures |   fail_rate |
|--------------:|----:|-----------:|------------:|
|             0 |  61 |          1 |       0.016 |
|             1 |  61 |          1 |       0.016 |
|             2 |  56 |          4 |       0.071 |
|             3 |  77 |         13 |       0.169 |

## Object, non-keys prompts only
chi2 = 13.02, permutation p = 0.0401

| object      |   n |   failures |   fail_rate |
|:------------|----:|-----------:|------------:|
| a book      |  37 |          5 |       0.135 |
| a card      |  43 |          0 |       0     |
| a drink     |  33 |          2 |       0.061 |
| the bag     |  36 |          0 |       0     |
| the ball    |  30 |          2 |       0.067 |
| the flowers |  39 |          4 |       0.103 |
| the gift    |  37 |          6 |       0.162 |

## Article (the/a), non-keys prompts only
chi2 = 0.46, permutation p = 0.6314

| article   |   n |   failures |   fail_rate |
|:----------|----:|-----------:|------------:|
| a         | 113 |          7 |       0.062 |
| the       | 142 |         12 |       0.085 |

## Failure modes (predicted token) among non-keys failures, by template

|   template_id |   her |   his |   the |
|--------------:|------:|------:|------:|
|             0 |     0 |     0 |     1 |
|             1 |     0 |     1 |     0 |
|             2 |     1 |     2 |     1 |
|             3 |    10 |     2 |     1 |

## Most failure-concentrated non-keys template: 3
13/77 fail vs 6/178 in the other templates (Fisher exact p = 0.0004013). Template 3: `Once Susan and Dan arrived at the park, Susan passed the keys to`

## Template texts

* 0: `When Lily and Chris went to the gym, Chris gave the keys to`
* 1: `After Karen and James went to the store, Karen gave the gift to`
* 2: `While James and Alex were at the gym, Alex handed a card to`
* 3: `Once Susan and Dan arrived at the park, Susan passed the keys to`
