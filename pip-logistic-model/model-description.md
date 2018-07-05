## Problem 1

## Problem 2

### Model 1

Tn X (N points) => M sampled points => DL (input : [Tn, L, 2], output : [1])

### Model 2

Tn X (N points) => M x M painted pixels ( 0 : blank, 1 : boundary, 2 : target ) => DL (input : [Tn, M, M, 1], output : [1])

One model - one vertices num

### Model 3

We don't care about how many sides in one polygon

Only convex hull data will be handled

#### 01

data store in memory

#### 02

batch - using pipeline from file

#### 03

more deep tensors

### Model 4

Not only convex hull data but simple polygon. Each polygon data has own convex hull ratio..