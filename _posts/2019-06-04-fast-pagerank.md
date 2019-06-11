---
layout: post
title:  "Fast PageRank Implementation in Python"
date:   2019-06-04
mathjax: true
---

# Fast Personalized PageRank Python Implementation

I needed a fast PageRank for [Wikisim](https://github.com/asajadi/wikisim) project. It had to be fast enough to run real time on relatively large graphs. NetworkX was the obvious library to use, however, it needed back and forth translation from my graph representation (which was the pretty standard csr matrix), to its internal graph data structure. These translations were slowing down the process.

I implemented two versions of the algorithm in Python, both inspired by the sparse fast solutions given in [**Cleve Moler**](https://en.wikipedia.org/wiki/Cleve_Moler)'s book, [*Experiments with MATLAB*](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/pagerank.pdf). The power method is much faster with enough precision for our task.

### Personalized PageRank
I modified the algorithm a little bit to be able to calculate **personalized PageRank** as well.


### Comparison with Popular Python Implementations: NetworkX and iGraph
Both implementations (exact solution and *power method*) are much faster than their correspondent methods in NetworkX. The *power method* is also faster than the iGraph native implementation, which is also an eigenvector based solution. Benchmarking is done on a `ml.t3.2xlarge` SageMaker instance.

### What is the major drawback of NetworkX PageRank?
I gave up using NetworkX for one simple reason: I had to calculate PageRank several times, and my internal representation of a graph was a simple sparse matrix. Every time I wanted to calculate PageRank I had to translate it to the graph representation of NetworkX, which was slow. My benchmarking shows that NetworkX  has a pretty fast implementation of PageRank ( `networkx.pagerank_numpy` and  '`networkx.pagerank_scipy`), but translating from its own graph data structure to a csr matrix before doing the actual calculations is exactly what exactly slows down the whole algorithm.

**Note**: I didn't count the time spent on `nx.from_scipy_sparse_matrix` (converting a csr matrix before passing it to NetworkX PageRank) in my benchmarking, But I could! Because that was another bottleneck for me, and for many other cases that one has a `csr` adjacency matrix.

### Python Implementation
The python package is hosted at https://github.com/asajadi/fast-pagerank and you can find the installation guide in the [README.md](https://github.com/asajadi/fast-pagerank#usage) file. You also can find this jupyter notebook in [the notebook directory](https://github.com/asajadi/fast-pagerank/blob/master/notebooks/Fast-PageRank.ipynb).


## Appendix

### What is Google PageRank Algorithm?
PageRank is another link analysis algorithm primarily used to rank search engine results. It is defined as a process in which starting  from a
random node, a random walker moves to a	random neighbour with probability $\alpha$  or jumps to a random vertex with the probability $1-\alpha$ . The PageRank values are the limiting probabilities of finding a walker on each
node. In the original PageRank, the jump can be to any node with a uniform probability, however later in **Personalized PageRank**, this can be any custom probability distribution over the nodes.

### How  Google PageRank is Calculated? [1, 2]

Let $\mathbf{A}$ be the adjacency matrix ($\mathbf{A}_{ij}$ is the weight of the edge from node $i$ to node $j$) and $\vec{s}$ be the *teleporting probability*, that is $\vec{s}_i$ is the probability of jumping to node $i$. Probability of being at node $j$ at time $t+1$  can be determined by two factors:
1. Sum over the out-neighbors $i$ of $j$ of the probability that the walk was at $i$ at time t, times the probability it moved from $i$ to $j$ in time $t+1$.
2. Probability of teleporting from somewhere else in the graph to $j$.

Formally:

\begin{equation}
    \vec{p}_{t+1}(j)=\alpha\sum_{i:(i,j)\in E}\frac{A(i,j)}{d(i)}\vec{p}_t(i)+(1-\alpha)\vec{s}_j,
\end{equation}

where $d(i)$ is the out-degree of node $i$.
To give a matrix form, we define $\mathbf{D}$ to be the diagonal matrix with the out-degree  of each node in $\mathbf{A}$ on
the diagonal. Then the PageRank
vector, initialized with $\vec{s}$, can be obtained from the following recursion:

\begin{equation}
    \vec{pr}_{t+1}=\alpha \mathbf{A}^T \mathbf{D}^{-1}\vec{pr}_{t}+(1-\alpha)\vec{s}.
\end{equation}

There is a serious problem that we need to take care: $\mathbf{D}^{-1}$ is the inverse of $\mathbf{D}$, which for a diagonal matrix it will be simply inverting the elements on the diagonal. This will break if there are nodes with no out neighbors, a.k.a, *dangling nodes*.
What happens when you hit a page with no out link? You only have one option and that is to jump to a random page.

To simulate this behavior we alter $\mathbf{A}$ by adding an edge from every dangling node to every other node $j$ with a weight of $\vec{s}_j$. In other words, we create $\mathbf{\bar{A}}$ by replacing each all zero row by $\vec{s}^T$. Formally, if we define $\vec{r}$ to be the vector of row-wise sum of the elements of $\mathbf{A}$, that is $\vec{r}_i=\sum_{j}A_{ij}$, then:

\begin{align}
\mathbf{\bar{A}}&=\mathbf{A}+\mathbf{B}\\
\mbox{where}\\
\mathbf{B}_{ij} &= \begin{cases}
                        \vec{s}_j & \mbox{if } r_i=0 \\
                        0   & \mbox{else}
                    \end{cases} \\  
\end{align}

We need to re-define $\mathbf{D}$. In our new definition of $\mathbf{D}$, we ignore nodes with no out-neighbors (or in other words, replace $\frac{1}{0}$ by $0$). Similar to $\mathbf{D}$, we define $\mathbf{\bar{D}}$ to be the diagonal matrix of the out-degrees of $\mathbf{\bar{A}}$. So we can rewrite the recursion as:

\begin{equation}
    \vec{pr}_{t+1}=\alpha \mathbf{\bar{A}}^T \mathbf{\bar{D}}^{-1}\vec{pr}_{t}+(1-\alpha)\vec{s}. \tag{I}\label{I}
\end{equation}

Now $\vec{pr}$, the stationary probabilities (i.e, when $\vec{pr}_{t+1}=\vec{pr}_t=\vec{pr}$) can be calculated by either of the following approaches:

**1. Linear System Solving**

We can solve Eq. $\eqref{I}$ and get:

\begin{equation}
    \vec{pr}=(I-\alpha\mathbf{\bar{A}}^T \mathbf{\bar{D}}^{-1})(1-\alpha)\vec{s}.
\end{equation}

And use a linear system solver to calculate $\vec{pr}$.

**2. Power-Method**

Basically, reiterating the Eq. $\eqref{I}$ until it converges.


### How Fast Google PageRank Is Calculated? [3]
To speed up, we need to take advantage of sparse matrix calculations.  The only problem with the current formulation is that $\mathbf{\bar{A}}$ has a lower sparsity than the original $\mathbf{A}$. However, we can move around pieces of the equation a little bit to skip forming this matrix. We know that:

\begin{align}
\mathbf{\bar{A}}^T \mathbf{\bar{D}}
                       &= (\mathbf{A}^T+\mathbf{B}^T)\mathbf{\bar{D}}\\
                       &= \mathbf{A}^T\mathbf{\bar{D}}^{-1}
                       +\mathbf{B}^T\mathbf{\bar{D}}^{-1}
\end{align}

For the first term, multiplying by this diagonal matrix scales each column and $\mathbf{\bar{D}}$ and $\mathbf{D}$ are different only in the elements whose correspondent columns were all zero in $\mathbf{A}^T$, so we can safely replace $\mathbf{\bar{D}}$ with $\mathbf{D}$. Also  $\mathbf{B}^T\mathbf{\bar{D}}^{-1}=\mathbf{B}^T$ because the non zero columns of $\mathbf{B}^T$ are all $\vec{s}$, which add up to $1$, and therefore their correspondent element on $\mathbf{D}$ will be $1$. Therefore,

\begin{align}
\mathbf{\bar{A}}^T \mathbf{\bar{D}}
                       &= \mathbf{A}^T\mathbf{D}^{-1}
                       +\mathbf{B}^T,
\end{align}


and using the above equation we can rewrite Eq. $\eqref{I}$ and get

\begin{align}
    \vec{pr}_{t+1} &= \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_{t}
                    +\alpha\mathbf{B}^T\vec{pr}_{t}
                    +(1-\alpha)\vec{s}. \tag{II}\label{II}
\end{align}

This recursion has three multiplications, and the last one is a rather expensive one ($\mathbf{B}$ is a $n\times n$ matrix, therefore the whole multiplication will be $O(n^2)$).

Being a normalized vector, we know that $\vec{1}^T\vec{pr}_t=1$. We can multiply the last term of Eq. $\eqref{II}$ with $\vec{1}^T\vec{pr}_t$ and factor out $\vec{pr}$:

\begin{align}
    \vec{pr} &=  \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_t
                  +\alpha\mathbf{B}^T\vec{pr}_t
                  +(1-\alpha)\vec{s}\vec{1}^T\vec{pr}_t \\
                   &= \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_t+
                    (\alpha\mathbf{B}^T+
                    (1-\alpha)\vec{s}\vec{1}^T)\vec{pr}_t. \tag{III}\label{III}
\end{align}

Let $\mathbf{C}$ be $\alpha\mathbf{B}^T+(1-\alpha)\vec{s}\vec{1}^T$. Notice that $\vec{s}\vec{1}^T$ is a matrix with $\vec{s}$ as its columns, and substituting the definition of $\mathbf{B}$, the matrix $\mathbf{C}$ will be:

\begin{align}
\mathbf{C}_{ij} &= \begin{cases}
                        \vec{s}_i & \mbox{if } r_j=0 \\
                        (1-\alpha)\vec{s}_i & \mbox{else}
                \end{cases} \\  
\end{align}

If we let $\vec{z}$ be:

\begin{align}
\vec{z}_i &= \begin{cases}
                1 & \mbox{if } r_i=0 \\
                (1-\alpha) & \mbox{else}
                \end{cases}  
\end{align}

then

\begin{equation}
\mathbf{C}=\vec{s}\vec{z}^T
\end{equation}

So by replacing  ($\alpha\mathbf{B}^T+(1-\alpha)\vec{s}\vec{1}^T$) in Eq. $\eqref{III}$ with $\vec{s}\vec{z}^T$, we'll get:

\begin{align}
    \vec{pr}_{t+1} &= \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_{t}+(\vec{s}\vec{z}^T)\vec{pr}_{t}. \tag{IV}\label{IV}  
\end{align}

How does this help to improve the calculations? We'll see:

**1. Solving a Linear System**

Similar to before, we can solve Eq. $\eqref{IV}$ and get:

\begin{equation}
    \vec{pr}=(I-\alpha \mathbf{A}^T\mathbf{D}^{-1})^{-1}(\vec{s}\vec{z}^T)\vec{pr}.
\end{equation}

Being able to re-parenthesize, $\vec{z}^T\vec{p}$ is just a number, so we can ignore it and renormalize $\vec{pr}$ at the end, and solve:

\begin{equation}
    \vec{pr}=(I-\alpha \mathbf{A}^T\mathbf{D}^{-1})^{-1}\vec{s}.
\end{equation}

We almost have the same linear equation system that we had before, except for one big improvement, we replaced the less-sparse $\mathbf{\bar{A}}$ with $\mathbf{A}$.

**2. Power Method**

We can apply one last smart modification to Eq. $\eqref{IV}$: if we change the parenthesizing of the last multiplication ([remember the famous dynamic programming algorithm](https://en.wikipedia.org/wiki/Matrix_chain_multiplication)?), and also define $\mathbf{W}=\alpha\mathbf{A}^T\mathbf{D}^{-1}$, we will have:

\begin{equation}
\vec{pr}_{t+1} = \mathbf{W}\vec{pr}_{t}+
                \vec{s}(\vec{z}^T\vec{pr}_{t})
\end{equation}

Therefore, the complexity decreased to $O(n)$, and the whole recursion will be $O(n)\times \#iterations$. The rate of convergence is another thing, which we ignore here, and depends on the value of the second eigenvalue ($\lambda_2$) of the modified transition matrix ($\mathbf{T}$), which is defined as:
\begin{equation}
\mathbf{T}=\alpha\mathbf{A}^T\mathbf{D}^{-1}+\vec{s}\vec{z}^T
\end{equation}


## References

[1] [Daniel A. Spielman](https://en.wikipedia.org/wiki/Daniel_Spielman), Graphs and Networks Lecture Notes, [Lecture 11: Cutting Graphs, Personal PageRank and Spilling Paint](http://www.cs.yale.edu/homes/spielman/462/lect11-13.pdf), 2013.

[2] [Daniel A. Spielman](https://en.wikipedia.org/wiki/Daniel_Spielman), Spectral Graph Theory Lecture Notes, [Lecture 10: Random Walks on Graphs](http://www.cs.yale.edu/homes/spielman/561/lect10-18.pdf), 2018

[3] [Cleve Moler](https://en.wikipedia.org/wiki/Cleve_Moler), *Experiments with MATLAB*, [Chapter 7:
Google PageRank](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/pagerank.pdf)


## Implementation


```python
%%writefile ../fast_pagerank/fast_pagerank.py
"""Two fast implementations of PageRank:
    An exact solution using a sparse linear system solver,
    and an a power method approximation.
    Both solutions are taking full advantage of sparse matrix calculations.

    [Reference]:
    Cleve Moler. 2011. Experiments with MATLAB (Electronic ed.).
    MathWorks, Inc.
"""
# uncomment
from __future__ import division

import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 2015, The Wikisim Project"
__email__ = "asajadi@gmail.com"


def pagerank(A, p=0.85,
             personalize=None, reverse=False):
    """ Calculates PageRank given a csr graph

    Inputs:
    -------

    G: a csr graph.
    p: damping factor
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically
    reverse: If true, returns the reversed-PageRank

    outputs
    -------

    PageRank Scores for the nodes

    """
    # In Moler's algorithm, $A_{ij}$ represents the existences of an edge
    # from node $j$ to $i$, while we have assumed the opposite!
    if reverse:
        A = A.T

    n, _ = A.shape
    r = sp.asarray(A.sum(axis=1)).reshape(-1)

    k = r.nonzero()[0]

    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = sp.ones(n)
    personalize = personalize.reshape(n, 1)
    s = (personalize / personalize.sum()) * n

    I = sprs.eye(n)
    x = sprs.linalg.spsolve((I - p * A.T @ D_1), s)

    x = x / x.sum()
    return x


def pagerank_power(A, p=0.85, max_iter=100,
                   tol=1e-06, personalize=None, reverse=False):
    """ Calculates PageRank given a csr graph

    Inputs:
    -------
    A: a csr graph.
    p: damping factor
    max_iter: maximum number of iterations
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically.
    reverse: If true, returns the reversed-PageRank

    Returns:
    --------
    PageRank Scores for the nodes

    """
    # In Moler's algorithm, $G_{ij}$ represents the existences of an edge
    # from node $j$ to $i$, while we have assumed the opposite!
    if reverse:
        A = A.T

    n, _ = A.shape
    r = sp.asarray(A.sum(axis=1)).reshape(-1)

    k = r.nonzero()[0]

    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = sp.ones(n)
    personalize = personalize.reshape(n, 1)
    s = (personalize / personalize.sum()) * n

    z_T = (((1 - p) * (r != 0) + (r == 0)) / n)[sp.newaxis, :]
    W = p * A.T @ D_1

    x = s
    oldx = sp.zeros((n, 1))

    iteration = 0

    while sp.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)

    return x.reshape(-1)
```

    Overwriting ../fast_pagerank/fast_pagerank.py


# Testing the algorithm


```python
%%writefile ../test/fast_pagerank_test.py
import os
import sys
import scipy as sp
import scipy.sparse as sparse
from numpy.testing import assert_allclose
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))

from fast_pagerank import pagerank
from fast_pagerank import pagerank_power


class TestMolerPageRank(unittest.TestCase):
    def setUp(self):
        # ---G1---
        n1 = 5
        edges1 = sp.array([[0, 1],
                           [1, 2],
                           [2, 1],
                           [2, 3],
                           [2, 4],
                           [3, 0],
                           [3, 2],
                           [4, 0],
                           [4, 2],
                           [4, 3]])
        weights1 = [0.4923,
                    0.0999,
                    0.2132,
                    0.0178,
                    0.5694,
                    0.0406,
                    0.2047,
                    0.8610,
                    0.3849,
                    0.4829]

        self.p1 = 0.83
        self.personalize1 = sp.array([0.6005, 0.1221, 0.2542, 0.4778, 0.4275])
        self.G1 = sparse.csr_matrix(
            (weights1, (edges1[:, 0], edges1[:, 1])), shape=(n1, n1))
        self.pr1 = sp.array([0.1592, 0.2114, 0.3085, 0.1, 0.2208])

        # ---G2---
        n2 = 10
        edges2 = sp.array([[2, 4],
                           [2, 5],
                           [4, 5],
                           [5, 3],
                           [5, 4],
                           [5, 9],
                           [6, 1],
                           [6, 2],
                           [9, 2],
                           [9, 4]])
        weights2 = [0.4565,
                    0.2861,
                    0.5730,
                    0.0025,
                    0.4829,
                    0.3866,
                    0.3041,
                    0.3407,
                    0.2653,
                    0.8079]
        self.G2 = sparse.csr_matrix(
            (weights2, (edges2[:, 0], edges2[:, 1])), shape=(n2, n2))
        self.personalize2 = sp.array([0.8887, 0.6491, 0.7843, 0.7103, 0.7428,
                                      0.6632, 0.7351, 0.3006, 0.8722, 0.1652])
        self.p2 = 0.92
        self.pr2 = sp.array([0.0234, 0.0255, 0.0629, 0.0196, 0.3303,
                             0.3436, 0.0194, 0.0079, 0.023, 0.1445])

        # ---G3---
        n3 = 5
        edges3 = sp.array([[2, 4]])
        weights3 = [0.5441]
        self.G3 = sparse.csr_matrix(
            (weights3, (edges3[:, 0], edges3[:, 1])), shape=(n3, n3))

        self.personalize3 = sp.array([0.0884, 0.2797, 0.3093, 0.5533, 0.985])
        self.p3 = 0.81
        self.pr3 = sp.array([0.0358, 0.1134, 0.1254, 0.2244, 0.501])

        # ---G4---
        n4 = 5
        edges4_rows = []
        edges4_cols = []
        weights4 = []
        self.G4 = sparse.csr_matrix(
            (weights4, (edges4_rows, edges4_cols)), shape=(n4, n4))

        self.personalize4 = sp.array([0.2534, 0.8945, 0.9562, 0.056, 0.9439])
        self.p4 = 0.70
        self.pr4 = sp.array([0.0816, 0.2882, 0.3081, 0.018, 0.3041])

        # ---G5---
        n5 = 0
        edges5_rows = []
        edges5_cols = []
        weights5 = []
        self.G5 = sparse.csr_matrix(
            (weights5, (edges5_rows, edges5_cols)), shape=(n5, n5))

        self.personalize5 = sp.array([])
        self.p5 = 0.70
        self.pr5 = sp.array([])

    def test_pagerank_1(self):
        calculated_pagerank = pagerank(self.G1, p=self.p1,
                                       personalize=self.personalize1)
        assert_allclose(calculated_pagerank, self.pr1, rtol=0, atol=1e-04)

    def test_pagerank_2(self):

        calculated_pagerank = pagerank(self.G2, p=self.p2,
                                       personalize=self.personalize2)
        assert_allclose(calculated_pagerank, self.pr2, rtol=0, atol=1e-04)

    def test_single_edge(self):
        calculated_pagerank = pagerank(self.G3, p=self.p3,
                                       personalize=self.personalize3)
        assert_allclose(calculated_pagerank, self.pr3, rtol=0, atol=1e-04)

    def test_zero_edge(self):
        calculated_pagerank = pagerank(self.G4, p=self.p4,
                                       personalize=self.personalize4)
        assert_allclose(calculated_pagerank, self.pr4, rtol=0, atol=1e-04)

    def test_empty_graph(self):
        calculated_pagerank = pagerank(self.G5, p=self.p5,
                                       personalize=self.personalize5)
        self.assertEqual(calculated_pagerank.size, 0)

    def test_power_pagerank_1(self):
        calculated_pagerank = pagerank_power(self.G1, p=self.p1,
                                             personalize=self.personalize1)
        assert_allclose(calculated_pagerank, self.pr1, rtol=0, atol=1e-04)

    def test_power_pagerank_2(self):

        calculated_pagerank = pagerank_power(self.G2, p=self.p2,
                                             personalize=self.personalize2)
        assert_allclose(calculated_pagerank, self.pr2, rtol=0, atol=1e-04)

    def test_power_single_edge(self):
        calculated_pagerank = pagerank_power(self.G3, p=self.p3,
                                             personalize=self.personalize3)
        assert_allclose(calculated_pagerank, self.pr3, rtol=0, atol=1e-04)

    def test_power_zero_edge(self):
        calculated_pagerank = pagerank_power(self.G4, p=self.p4,
                                             personalize=self.personalize4)
        assert_allclose(calculated_pagerank, self.pr4, rtol=0, atol=1e-04)

    def test_power_empty_graph(self):
        calculated_pagerank = pagerank_power(self.G5, p=self.p5,
                                             personalize=self.personalize5)
        self.assertEqual(calculated_pagerank.size, 0)


#             assert_array_almost_equal(Ynx,  Yml, decimal = 5)
if __name__ == '__main__':
    unittest.main()
```

    Overwriting ../test/fast_pagerank_test.py



```python
!python  ../test/fast_pagerank_test.py
```

    /Users/arminsajadi/anaconda3/lib/python3.7/site-packages/numpy/matrixlib/defmatrix.py:71: PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
      return matrix(data, dtype=dtype, copy=False)
    ..........
    ----------------------------------------------------------------------
    Ran 10 tests in 0.020s

    OK


# Benchmarking

To avoid the clutter, we only visualize the fastest method from each implementation, that is:

- `networkx.pagerank_scipy`
- Latest implementation of  `iGraph.personalized_pagerank` (PRPACK)
- Our `pagerank_power`



```python
''' Calcualate PageRank on several random graphs.
'''
import scipy as sp
import pandas as pd
import timeit
import os
import sys
import random
import igraph
import networkx as nx
sys.path.insert(0, '..')
from fast_pagerank.pagerank import pagerank
from fast_pagerank.pagerank import pagerank_power

# def print_and_flush(args):

#     sys.stdout.flush()
def get_random_graph(
        min_size=20,
        max_size=2000,
        min_density=0.1,
        max_density=0.5):
    ''' Creates a random graph and a teleport vector and output them
        in different formats for different algorithms

    Inputs
    ------

    min_size and max_size: The size of the graph will be a random number
        in the range of (min_size, max_size)
    min_sparsity and max_sparsity: The sparcity of the graph
        will be a random number in the range of (min_sparsity, max_sparsity)

    Returns
    -------

    nxG: A random Graph for NetworkX
    A: The equivallent csr Adjacency matrix, for our PageRank
    iG: The equivallent iGraph
    personalize_vector: Personalization probabily vector
    personalize_dict: Personalization probabily vector,
                    in the form of a dictionary for NetworkX

    '''
    G_size = random.randint(min_size, max_size)
    p = random.uniform(min_density, max_density)

    A = sp.sparse.random(G_size, G_size, p, format='csr')
    nxG = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())

    iG = igraph.Graph(list(nxG.edges()), directed=True)
    iG.es['weight'] = A.data

    personalize_vector = sp.random.random(G_size)
    personalize_dict = dict(enumerate(personalize_vector.reshape(-1)))
    return A, nxG, iG, personalize_vector, personalize_dict


n = 5
number_of_graphs = 50

node_size_vector = sp.zeros(number_of_graphs)
edge_size_vector = sp.zeros(number_of_graphs)
# netx_pagerank_times = sp.zeros(number_of_graphs)
netx_pagerank_times_numpy = sp.zeros(number_of_graphs)
netx_pagerank_times_scipy = sp.zeros(number_of_graphs)
ig_pagerank_times = sp.zeros(number_of_graphs)
pagerank_times = sp.zeros(number_of_graphs)
pagerank_times_power = sp.zeros(number_of_graphs)

damping_factor = 0.85
tol = 1e-3


for i in range(number_of_graphs):
    A, nxG, iG, personalize_vector, personalize_dict = get_random_graph()
    node_size_vector[i] = A.shape[0]
    edge_size_vector[i] = A.count_nonzero()
    print ("Graph %d: Nodes: %d, Edges: %d ..." %(i, node_size_vector[i], edge_size_vector[i]))
    sys.stdout.flush()

#     networkx.pagerank commented out, because it is too slow

#     netx_pagerank_times[i] = timeit.timeit(
#         lambda: nx.pagerank(nxG, alpha=damping_factor, tol=tol),
#         number=n) / n

    netx_pagerank_times_numpy[i] = timeit.timeit(
        lambda: nx.pagerank_numpy(nxG, alpha=damping_factor),
        number=n) / n

    netx_pagerank_times_scipy[i] = timeit.timeit(
        lambda: nx.pagerank_scipy(nxG, alpha=damping_factor, tol=tol),
        number=n) / n

    #iGraph, only "prpack", which is their latest version.
    ig_pagerank_times[i] = timeit.timeit(
        lambda: iG.personalized_pagerank(directed=True,
                                         damping=damping_factor,
                                         weights=iG.es['weight'],
                                         implementation="prpack"),
        number=n) / n

#     My implementations

    pagerank_times[i] = timeit.timeit(
        lambda: pagerank(A, p=damping_factor),
        number=n) / n
    pagerank_times_power[i] = timeit.timeit(
        lambda: pagerank_power(A, p=damping_factor, tol=tol),
        number=n) / n


argsort = edge_size_vector.argsort()

edge_size_vector_sorted = edge_size_vector[argsort]
node_size_vector_sorted = node_size_vector[argsort]

# netx_pagerank_times_sorted = netx_pagerank_times[argsort]
netx_pagerank_times_numpy_sorted = netx_pagerank_times_numpy[argsort]
netx_pagerank_times_scipy_sorted = netx_pagerank_times_scipy[argsort]

ig_pagerank_times_sorted = ig_pagerank_times[argsort]

pagerank_times_sorted = pagerank_times[argsort]
pagerank_times_power_sorted = pagerank_times_power[argsort]

comparison_table = pd.DataFrame(list(zip(node_size_vector_sorted,
                                         edge_size_vector_sorted,
#                                          netx_pagerank_times_sorted,
                                         netx_pagerank_times_numpy_sorted,
                                         netx_pagerank_times_scipy_sorted,
                                         ig_pagerank_times_sorted,
                                         pagerank_times_sorted,
                                         pagerank_times_power_sorted)),
                                columns=['Nodes', 'Edges',
#                                          'NetX',
                                         'NetX (numpy)',
                                         'NetX (scipy)',
                                         'iGraph',
                                         '(fast) pagerank',
                                         '(fast) pagerank_power']).\
                    astype({'Nodes': 'int32', 'Edges': 'int32'})
comparison_table.to_csv('pagerank_methods_comparison.csv')
print("Done")
```

# Plotting


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

comparison_table = pd.read_csv('pagerank_methods_comparison.csv', index_col=0)
display(comparison_table)

plt.ioff()
fig=plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')


# plt.plot(comparison_table['Edges'], comparison_table['NetX'],
#          'o-', ms=8, lw=2, alpha=0.7, color='cyan',
#          label='networkx.PageRank')
plt.plot(comparison_table['Edges'], comparison_table['NetX (numpy)'],
         'v-', ms=8, lw=2, alpha=0.7, color='magenta',
         label='networkx.PageRank_numpy')

plt.plot(comparison_table['Edges'], comparison_table['NetX (scipy)'],
         'P-', ms=8, lw=2, alpha=0.7, color='blue',
         label='networkx.PageRank_scipy')

plt.plot(comparison_table['Edges'], comparison_table['iGraph'],
         'x-', ms=8, lw=2, alpha=0.7, color='black',
         label='iGraph_PageRank_ARPACK')

plt.plot(comparison_table['Edges'], comparison_table['(fast) pagerank'],
         '*-', ms=8, lw=2, alpha=0.7, color='red',
         label='fast_pagerank.pagerank')

plt.plot(comparison_table['Edges'], comparison_table['(fast) pagerank_power'],
         '^-', ms=8, lw=2, alpha=0.7, color='green',
         label='fast_pagerank.pagerank_power')


plt.xlabel('Number of the edges')
plt.ylabel('Time (Seconds)')


plt.tight_layout()
plt.legend(loc=2)
plt.savefig('pagerank_methods_comparison.png')
plt.show()

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nodes</th>
      <th>Edges</th>
      <th>NetX (numpy)</th>
      <th>NetX (scipy)</th>
      <th>iGraph</th>
      <th>(fast) pagerank</th>
      <th>(fast) pagerank_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29</td>
      <td>294</td>
      <td>0.001380</td>
      <td>0.002786</td>
      <td>0.000064</td>
      <td>0.001197</td>
      <td>0.000987</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>297</td>
      <td>0.002001</td>
      <td>0.002855</td>
      <td>0.000116</td>
      <td>0.001359</td>
      <td>0.001350</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57</td>
      <td>511</td>
      <td>0.003088</td>
      <td>0.004122</td>
      <td>0.000207</td>
      <td>0.001339</td>
      <td>0.001040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1268</td>
      <td>0.005002</td>
      <td>0.008682</td>
      <td>0.000302</td>
      <td>0.001361</td>
      <td>0.000923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>153</td>
      <td>4239</td>
      <td>0.025173</td>
      <td>0.026826</td>
      <td>0.001618</td>
      <td>0.002090</td>
      <td>0.001170</td>
    </tr>
    <tr>
      <th>5</th>
      <td>161</td>
      <td>7313</td>
      <td>0.034990</td>
      <td>0.045815</td>
      <td>0.002600</td>
      <td>0.002153</td>
      <td>0.001118</td>
    </tr>
    <tr>
      <th>6</th>
      <td>194</td>
      <td>13112</td>
      <td>0.059986</td>
      <td>0.079006</td>
      <td>0.004251</td>
      <td>0.002485</td>
      <td>0.001197</td>
    </tr>
    <tr>
      <th>7</th>
      <td>235</td>
      <td>23995</td>
      <td>0.100380</td>
      <td>0.144351</td>
      <td>0.007696</td>
      <td>0.003906</td>
      <td>0.001367</td>
    </tr>
    <tr>
      <th>8</th>
      <td>399</td>
      <td>24620</td>
      <td>0.152208</td>
      <td>0.147999</td>
      <td>0.008201</td>
      <td>0.009624</td>
      <td>0.001632</td>
    </tr>
    <tr>
      <th>9</th>
      <td>331</td>
      <td>28117</td>
      <td>0.138913</td>
      <td>0.167950</td>
      <td>0.009146</td>
      <td>0.006118</td>
      <td>0.001569</td>
    </tr>
    <tr>
      <th>10</th>
      <td>400</td>
      <td>31555</td>
      <td>0.170775</td>
      <td>0.189163</td>
      <td>0.010441</td>
      <td>0.009061</td>
      <td>0.001658</td>
    </tr>
    <tr>
      <th>11</th>
      <td>432</td>
      <td>34350</td>
      <td>0.193550</td>
      <td>0.206139</td>
      <td>0.011067</td>
      <td>0.010935</td>
      <td>0.002538</td>
    </tr>
    <tr>
      <th>12</th>
      <td>327</td>
      <td>40070</td>
      <td>0.172449</td>
      <td>0.236692</td>
      <td>0.013152</td>
      <td>0.006070</td>
      <td>0.001624</td>
    </tr>
    <tr>
      <th>13</th>
      <td>345</td>
      <td>43278</td>
      <td>0.185561</td>
      <td>0.257730</td>
      <td>0.013519</td>
      <td>0.006637</td>
      <td>0.001671</td>
    </tr>
    <tr>
      <th>14</th>
      <td>372</td>
      <td>51392</td>
      <td>0.217195</td>
      <td>0.306374</td>
      <td>0.016442</td>
      <td>0.007388</td>
      <td>0.001801</td>
    </tr>
    <tr>
      <th>15</th>
      <td>443</td>
      <td>53407</td>
      <td>0.257006</td>
      <td>0.318984</td>
      <td>0.016917</td>
      <td>0.010931</td>
      <td>0.001912</td>
    </tr>
    <tr>
      <th>16</th>
      <td>513</td>
      <td>53818</td>
      <td>0.332424</td>
      <td>0.322413</td>
      <td>0.017470</td>
      <td>0.017187</td>
      <td>0.002074</td>
    </tr>
    <tr>
      <th>17</th>
      <td>657</td>
      <td>58504</td>
      <td>0.431417</td>
      <td>0.349819</td>
      <td>0.019168</td>
      <td>0.029765</td>
      <td>0.002254</td>
    </tr>
    <tr>
      <th>18</th>
      <td>600</td>
      <td>66924</td>
      <td>0.424966</td>
      <td>0.427820</td>
      <td>0.021523</td>
      <td>0.022974</td>
      <td>0.002296</td>
    </tr>
    <tr>
      <th>19</th>
      <td>708</td>
      <td>68816</td>
      <td>0.493887</td>
      <td>0.412282</td>
      <td>0.023177</td>
      <td>0.035180</td>
      <td>0.002426</td>
    </tr>
    <tr>
      <th>20</th>
      <td>595</td>
      <td>70160</td>
      <td>0.447111</td>
      <td>0.421269</td>
      <td>0.022601</td>
      <td>0.021008</td>
      <td>0.002358</td>
    </tr>
    <tr>
      <th>21</th>
      <td>402</td>
      <td>72749</td>
      <td>0.286772</td>
      <td>0.432235</td>
      <td>0.022592</td>
      <td>0.009006</td>
      <td>0.002071</td>
    </tr>
    <tr>
      <th>22</th>
      <td>527</td>
      <td>103922</td>
      <td>0.487291</td>
      <td>0.621258</td>
      <td>0.032654</td>
      <td>0.015822</td>
      <td>0.002414</td>
    </tr>
    <tr>
      <th>23</th>
      <td>552</td>
      <td>113892</td>
      <td>0.511531</td>
      <td>0.682059</td>
      <td>0.035800</td>
      <td>0.018713</td>
      <td>0.002606</td>
    </tr>
    <tr>
      <th>24</th>
      <td>643</td>
      <td>121678</td>
      <td>0.608551</td>
      <td>0.733741</td>
      <td>0.038143</td>
      <td>0.025314</td>
      <td>0.003068</td>
    </tr>
    <tr>
      <th>25</th>
      <td>535</td>
      <td>131004</td>
      <td>0.548765</td>
      <td>0.780345</td>
      <td>0.041276</td>
      <td>0.017120</td>
      <td>0.002782</td>
    </tr>
    <tr>
      <th>26</th>
      <td>788</td>
      <td>134715</td>
      <td>0.738787</td>
      <td>0.805370</td>
      <td>0.043985</td>
      <td>0.040695</td>
      <td>0.002948</td>
    </tr>
    <tr>
      <th>27</th>
      <td>659</td>
      <td>139356</td>
      <td>0.655564</td>
      <td>0.832847</td>
      <td>0.044811</td>
      <td>0.027062</td>
      <td>0.002884</td>
    </tr>
    <tr>
      <th>28</th>
      <td>574</td>
      <td>142418</td>
      <td>0.614663</td>
      <td>0.851331</td>
      <td>0.044907</td>
      <td>0.019513</td>
      <td>0.002872</td>
    </tr>
    <tr>
      <th>29</th>
      <td>671</td>
      <td>211463</td>
      <td>0.860593</td>
      <td>1.270099</td>
      <td>0.066796</td>
      <td>0.028440</td>
      <td>0.003524</td>
    </tr>
    <tr>
      <th>30</th>
      <td>991</td>
      <td>217314</td>
      <td>1.195324</td>
      <td>1.296652</td>
      <td>0.069362</td>
      <td>0.074023</td>
      <td>0.003918</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1145</td>
      <td>224636</td>
      <td>1.376988</td>
      <td>1.351827</td>
      <td>0.072020</td>
      <td>0.115271</td>
      <td>0.004329</td>
    </tr>
    <tr>
      <th>32</th>
      <td>814</td>
      <td>240810</td>
      <td>1.044791</td>
      <td>1.447768</td>
      <td>0.076709</td>
      <td>0.044440</td>
      <td>0.004033</td>
    </tr>
    <tr>
      <th>33</th>
      <td>946</td>
      <td>329445</td>
      <td>1.435774</td>
      <td>1.980942</td>
      <td>0.108792</td>
      <td>0.063489</td>
      <td>0.004937</td>
    </tr>
    <tr>
      <th>34</th>
      <td>876</td>
      <td>368149</td>
      <td>1.467089</td>
      <td>2.206095</td>
      <td>0.122426</td>
      <td>0.054545</td>
      <td>0.005101</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1226</td>
      <td>382630</td>
      <td>1.951984</td>
      <td>2.299428</td>
      <td>0.128249</td>
      <td>0.125456</td>
      <td>0.005811</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1082</td>
      <td>414322</td>
      <td>1.791635</td>
      <td>2.493037</td>
      <td>0.140365</td>
      <td>0.093733</td>
      <td>0.005861</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1200</td>
      <td>463903</td>
      <td>2.105691</td>
      <td>2.787735</td>
      <td>0.158396</td>
      <td>0.122534</td>
      <td>0.007137</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1664</td>
      <td>482849</td>
      <td>3.057695</td>
      <td>2.898690</td>
      <td>0.166557</td>
      <td>0.295168</td>
      <td>0.008168</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1502</td>
      <td>488347</td>
      <td>2.682661</td>
      <td>2.949617</td>
      <td>0.168611</td>
      <td>0.227022</td>
      <td>0.007144</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1440</td>
      <td>621314</td>
      <td>2.905153</td>
      <td>3.730029</td>
      <td>0.219487</td>
      <td>0.196364</td>
      <td>0.008039</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1671</td>
      <td>629123</td>
      <td>3.468734</td>
      <td>3.764259</td>
      <td>0.223085</td>
      <td>0.310846</td>
      <td>0.008445</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1476</td>
      <td>800797</td>
      <td>3.515316</td>
      <td>4.820850</td>
      <td>0.298863</td>
      <td>0.221590</td>
      <td>0.009985</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1567</td>
      <td>858346</td>
      <td>3.868013</td>
      <td>5.151185</td>
      <td>0.319868</td>
      <td>0.276208</td>
      <td>0.010889</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1442</td>
      <td>920573</td>
      <td>3.741552</td>
      <td>5.525220</td>
      <td>0.349847</td>
      <td>0.202256</td>
      <td>0.010942</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1773</td>
      <td>1039771</td>
      <td>4.812556</td>
      <td>6.241451</td>
      <td>0.396200</td>
      <td>0.327685</td>
      <td>0.013697</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1845</td>
      <td>1089758</td>
      <td>5.186088</td>
      <td>6.565350</td>
      <td>0.413339</td>
      <td>0.391772</td>
      <td>0.013254</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1995</td>
      <td>1368626</td>
      <td>6.392365</td>
      <td>8.204485</td>
      <td>0.535340</td>
      <td>0.461299</td>
      <td>0.016423</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1748</td>
      <td>1489884</td>
      <td>6.044785</td>
      <td>8.972130</td>
      <td>0.608673</td>
      <td>0.353495</td>
      <td>0.016298</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1996</td>
      <td>1552304</td>
      <td>6.911417</td>
      <td>9.262027</td>
      <td>0.613982</td>
      <td>0.475427</td>
      <td>0.018804</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_9_1.png)
