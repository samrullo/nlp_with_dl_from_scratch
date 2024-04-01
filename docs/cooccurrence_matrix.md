# Issues with raw co-occurrence matrix
There is an issue with the raw co-occurrence we have built from the corpus.
That issue arises due to high frequency words. 
For instance word *the* and word *car*, might occur in a corpus many times.
As a result, even though we would expect higher relationship between words *car* and *drive*,
our co-occurrence matrix will tell us that *car* is more related to *the*, rather than *drive*

To solve this issue we use **Pointwise Mutual Information**, which is defined as below

$$
PMI(X,Y) = \log_2\frac{P(X,Y)}{P(X) \cdot P(Y)}
$$

- PMI can dampen the impact of high frequency words and give us more intuitive relatedness score between words
- The formula makes sense. Firs of all, PMI of X and Y is the probability of X and Y occurring together. However, if X or Y is a highly probable variable by itself, it will push PMI value down, which is what we expect. For instance *the* is a high probability event by itself in any corpus. As a result PMI between any word and *the* will always be pushed down by the denominator of **PMI**

- We can calculate **PMI** based on corpus. E.g. let X be word *the* and let Y be word *car*. And let *N* be the total number of words in the corpus. Then 

$$
P(X) = \frac{C(X)}{N}, P(Y)=\frac{C(Y)}{N}, P(X,Y) = \frac{C(X,Y)}{N}
$$

$$
PMI(X,Y) = \log_{2}\frac{\frac{C(X,Y)}{N}}{\frac{C(X)}{N}\cdot\frac{C(Y)}{N}}
$$

$$
PMI(X,Y) = \log_{2}\frac{N \cdot C(X,Y)}{C(X) \cdot C(Y)}
$$

- Let's use an example. Suppose we have a corpus consisting of 10,000 words. Next, let's say word *the* occurs 1000 times, word *car* occurs 20 times, *drive* occurs 10 times. Next let's say the words *the* and *car* co-occur 10 times and the words *car* and *drive* co-occur 5 times. Let's calculate what will be **PMI** of (*the*,*car*) and **PMI** of (*car*,*drive*)

$$
PMI(the,car) = \log_{2}\frac{10000 \cdot 10}{1000*20} = 2.32
$$

$$
PMI(car,drive) = \log_{2}\frac{10000 \cdot 5}{10*20} = 7.96
$$

From above we can confirm that **PPMI** correctly captures stronger relationship between *car* and *drive* and not get impacted by high frequency word *the*

Raw **PMI** has a small issue where it can generate a negative number. This may happen when two words occur very rarely together. And negative number doesn't make much sense as the score of relatedness for two words.
To solve this issue we use **PPMI** which stands for *Positive PMI*.

It is defined as 

$$
PPMI(X,Y) = max(0, PMI(X,Y))
$$

# Reduce dimension
When you build co-occurrence matrices as well as PPMI matrices, you will notice that these matrices are sparce. What it means is that for a give row, majority of the columns have zero values and only a few will have non-zero values. This wastes memory and computation when doing similarity search between words. Luckiliy we have techniques available to reduce the dimension of the vector representation of words based on PPMI matrix.

Dimensionality reduction reduces the number of columns dramatically while preserving as much valuable information as possible.

A simple example that can demonstrate dimensionality reduction is a linear regression. Imagine you have a data with two dimensions *(x,y)*. If *y* has strong linear relationship with *x* then we can express *y* in terms of *x* only. Thus we could express our dataset in terms of **1** dimension instead of **2**.

Benefits of dimensionality reduction

Dimensionality reduction can provide several key benefits in data processing and analysis, particularly when dealing with high-dimensional datasets. Here are three significant benefits:

### 1. Reducing Overfitting

High-dimensional datasets, where the number of features greatly exceeds the number of observations, can lead to overfitting in predictive modeling. Overfitting occurs when a model learns not only the underlying patterns in the data but also the noise specific to the training set, which negatively impacts the model’s performance on new, unseen data. Dimensionality reduction techniques can help mitigate this problem by simplifying the models. By reducing the number of features, there is less chance of the model picking up on the noise, and it can generalize better to new data.

### 2. Computational Efficiency

As the dimensionality of the dataset increases, the computational complexity of processing the data can grow exponentially—a phenomenon often referred to as the "curse of dimensionality." This can make the data analysis, visualization, and modeling processes very slow and resource-intensive. Dimensionality reduction can alleviate these issues by decreasing the number of features that need to be processed. This can lead to faster training times for machine learning models, quicker data processing, and less memory usage, making the analysis of large datasets more feasible.

### 3. Improved Data Visualization

Visualizing high-dimensional data is challenging because humans can typically only perceive data in two or three dimensions. Dimensionality reduction techniques can be used to project high-dimensional data into two or three dimensions while preserving as much of the significant structure of the data as possible. This makes it easier to visualize and identify patterns, trends, and outliers in the data. Techniques such as Principal Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE), and Uniform Manifold Approximation and Projection (UMAP) are often used for this purpose and can be incredibly powerful tools for exploratory data analysis.

In addition to these benefits, dimensionality reduction can also lead to better data storage and data transmission efficiency, and it can enable the use of algorithms that are not feasible with high-dimensional data. However, it's important to apply dimensionality reduction techniques carefully, as they can also result in the loss of important information if not used appropriately.