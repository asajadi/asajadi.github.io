---
layout: post
title:  "Fast PageRank Implementation in Python"
date:   2019-06-04
mathjax: true
comments: true
---

I needed a fast PageRank for [Wikisim](https://github.com/asajadi/wikisim) project. It had to be fast enough to run real time on relatively large graphs. NetworkX was the obvious library to use, however, it needed back and forth translation from my graph representation (which was the pretty standard csr matrix), to its internal graph data structure. These translations were slowing down the process.

I implemented two versions of the algorithm in Python, both inspired by the sparse fast solutions given in [**Cleve Moler**](https://en.wikipedia.org/wiki/Cleve_Moler)'s book, [*Experiments with MATLAB*](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/pagerank.pdf). The power method is much faster with enough precision for our task.

### Personalized PageRank
I modified the algorithm a little bit to be able to calculate **personalized PageRank** as well.


### Comparison with Popular Python Implementations: NetworkX and iGraph
Both implementations (exact solution and *power method*) are much faster than their correspondent methods in NetworkX. The *power method* is also faster than the iGraph native implementation, which is also an eigenvector based solution. Benchmarking is done on a machine with 4 cpus (2.70GHz) and  `16GB` of RAM.

### What is the major drawback of NetworkX PageRank?
I gave up using NetworkX for one simple reason: I had to calculate PageRank several times, and my internal representation of a graph was a simple sparse matrix. Every time I wanted to calculate PageRank I had to translate it to the graph representation of NetworkX, which was slow. My benchmarking shows that NetworkX  has a pretty fast implementation of PageRank, but translating from its own graph data structure to a csr matrix before doing the actual calculations is exactly what exactly slows down the whole algorithm.

**Note**: I didn't count the time spent on `nx.from_scipy_sparse_array` (converting a csr matrix before passing it to NetworkX PageRank) in my benchmarking, But I could! Because that was another bottleneck for me, and for many other cases that one has a `csr` adjacency matrix.

### Python Implementation
The python package is hosted at https://github.com/asajadi/fast-pagerank and you can find the installation guide in the [README.md](https://github.com/asajadi/fast-pagerank#usage) file. You also can find this jupyter notebook in [the notebook directory](https://github.com/asajadi/fast-pagerank/blob/master/notebooks/Fast-PageRank.ipynb).


## Appendix

### What is Google PageRank Algorithm?
PageRank is another link analysis algorithm primarily used to rank search engine results. It is defined as a process in which starting  from a
random node, a random walker moves to a	random neighbour with probability $$\alpha$$  or jumps to a random vertex with the probability $$1-\alpha$$ . The PageRank values are the limiting probabilities of finding a walker on each
node. In the original PageRank, the jump can be to any node with a uniform probability, however later in **Personalized PageRank**, this can be any custom probability distribution over the nodes.

### How  Google PageRank is Calculated? [1, 2]

Let $$\mathbf{A}$$ be the adjacency matrix ($$\mathbf{A}_{ij}$$ is the weight of the edge from node $$i$$ to node $$j$$) and $$\vec{s}$$ be the *teleporting probability*, that is $$\vec{s}_i$$ is the probability of jumping to node $$i$$. Probability of being at node $$j$$ at time $$t+1$$  can be determined by two factors:
1. Sum over the out-neighbors $$i$$ of $$j$$ of the probability that the walk was at $$i$$ at time t, times the probability it moved from $$i$$ to $$j$$ in time $$t+1$$.
2. Probability of teleporting from somewhere else in the graph to $$j$$.

Formally:

$$
\begin{equation}
    \vec{p}_{t+1}(j)=\alpha\sum_{i:(i,j)\in E}\frac{A(i,j)}{d(i)}\vec{p}_t(i)+(1-\alpha)\vec{s}_j,
\end{equation}
$$

where $$d(i)$$ is the out-degree of node $$i$$.
To give a matrix form, we define $$\mathbf{D}$$ to be the diagonal matrix with the out-degree  of each node in $$\mathbf{A}$$ on
the diagonal. Then the PageRank
vector, initialized with $$\vec{s}$$, can be obtained from the following recursion:

$$
\begin{equation}
    \vec{pr}_{t+1}=\alpha \mathbf{A}^T \mathbf{D}^{-1}\vec{pr}_{t}+(1-\alpha)\vec{s}.
\end{equation}
$$

There is a serious problem that we need to take care: $$\mathbf{D}^{-1}$$ is the inverse of $$\mathbf{D}$$, which for a diagonal matrix it will be simply inverting the elements on the diagonal. This will break if there are nodes with no out neighbors, a.k.a, *dangling nodes*.
What happens when you hit a page with no out link? You only have one option and that is to jump to a random page.

To simulate this behavior we alter $$\mathbf{A}$$ by adding an edge from every dangling node to every other node $$j$$ with a weight of $$\vec{s}_j$$. In other words, we create $$\mathbf{\bar{A}}$$ by replacing each all zero row by $$\vec{s}^T$$. Formally, if we define $$\vec{r}$$ to be the vector of row-wise sum of the elements of $$\mathbf{A}$$, that is $$\vec{r}_i=\sum_{j}A_{ij}$$, then:

$$
\begin{align}
\mathbf{\bar{A}}&=\mathbf{A}+\mathbf{B}\\
\mbox{where}\\
\mathbf{B}_{ij} &= \begin{cases}
                        \vec{s}_j & \mbox{if } r_i=0 \\
                        0   & \mbox{else}
                    \end{cases}
\end{align}
$$

We need to re-define $$\mathbf{D}$$. In our new definition of $$\mathbf{D}$$, we ignore nodes with no out-neighbors (or in other words, replace $$\frac{1}{0}$$ by $$0$$). Similar to $$\mathbf{D}$$, we define $$\mathbf{\bar{D}}$$ to be the diagonal matrix of the out-degrees of $$\mathbf{\bar{A}}$$. So we can rewrite the recursion as:

$$
\begin{equation}
    \vec{pr}_{t+1}=\alpha \mathbf{\bar{A}}^T \mathbf{\bar{D}}^{-1}\vec{pr}_{t}+(1-\alpha)\vec{s}. \tag{I}\label{I}
\end{equation}
$$

Now $$\vec{pr}$$, the stationary probabilities (i.e, when $$\vec{pr}_{t+1}=\vec{pr}_t=\vec{pr}$$) can be calculated by either of the following approaches:

**1. Linear System Solving**

We can solve Eq. $$\eqref{I}$$ and get:

$$
\begin{equation}
    \vec{pr}=(I-\alpha\mathbf{\bar{A}}^T \mathbf{\bar{D}}^{-1})(1-\alpha)\vec{s}.
\end{equation}
$$

And use a linear system solver to calculate $$\vec{pr}$$.

**2. Power-Method**

Basically, reiterating the Eq. $$\eqref{I}$$ until it converges.


### How Fast Google PageRank Is Calculated? [3]
To speed up, we need to take advantage of sparse matrix calculations.  The only problem with the current formulation is that $$\mathbf{\bar{A}}$$ has a lower sparsity than the original $$\mathbf{A}$$. However, we can move around pieces of the equation a little bit to skip forming this matrix. We know that:

$$
\begin{align}
\mathbf{\bar{A}}^T \mathbf{\bar{D}}
                       &= (\mathbf{A}^T+\mathbf{B}^T)\mathbf{\bar{D}}\\
                       &= \mathbf{A}^T\mathbf{\bar{D}}^{-1}
                       +\mathbf{B}^T\mathbf{\bar{D}}^{-1}
\end{align}
$$

For the first term, multiplying by this diagonal matrix scales each column and $$\mathbf{\bar{D}}$$ and $$\mathbf{D}$$ are different only in the elements whose correspondent columns were all zero in $$\mathbf{A}^T$$, so we can safely replace $$\mathbf{\bar{D}}$$ with $$\mathbf{D}$$. Also  $$\mathbf{B}^T\mathbf{\bar{D}}^{-1}=\mathbf{B}^T$$ because the non zero columns of $$\mathbf{B}^T$$ are all $$\vec{s}$$, which add up to $$1$$, and therefore their correspondent element on $$\mathbf{D}$$ will be $$1$$. Therefore,

$$
\begin{align}
\mathbf{\bar{A}}^T \mathbf{\bar{D}}
                       &= \mathbf{A}^T\mathbf{D}^{-1}
                       +\mathbf{B}^T,
\end{align}
$$


and using the above equation we can rewrite Eq. $$\eqref{I}$$ and get

$$
\begin{align}
    \vec{pr}_{t+1} &= \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_{t}
                    +\alpha\mathbf{B}^T\vec{pr}_{t}
                    +(1-\alpha)\vec{s}. \tag{II}\label{II}
\end{align}
$$

This recursion has three multiplications, and the last one is a rather expensive one ($$\mathbf{B}$$ is a $$n\times n$$ matrix, therefore the whole multiplication will be $$O(n^2)$$).

Being a normalized vector, we know that $$\vec{1}^T\vec{pr}_t=1$$. We can multiply the last term of Eq. $$\eqref{II}$$ with $$\vec{1}^T\vec{pr}_t$$ and factor out $$\vec{pr}$$:

$$
\begin{align}
    \vec{pr} &=  \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_t
                  +\alpha\mathbf{B}^T\vec{pr}_t
                  +(1-\alpha)\vec{s}\vec{1}^T\vec{pr}_t \\
                   &= \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_t+
                    (\alpha\mathbf{B}^T+
                    (1-\alpha)\vec{s}\vec{1}^T)\vec{pr}_t. \tag{III}\label{III}
\end{align}
$$

Let $$\mathbf{C}$$ be $$\alpha\mathbf{B}^T+(1-\alpha)\vec{s}\vec{1}^T$$. Notice that $$\vec{s}\vec{1}^T$$ is a matrix with $$\vec{s}$$ as its columns, and substituting the definition of $$\mathbf{B}$$, the matrix $$\mathbf{C}$$ will be:

$$
\begin{align}
\mathbf{C}_{ij} &= \begin{cases}
                        \vec{s}_i & \mbox{if } r_j=0 \\
                        (1-\alpha)\vec{s}_i & \mbox{else}
                    \end{cases}
\end{align}
$$

If we let $$\vec{z}$$ be:

$$
\begin{align}
\vec{z}_i &= \begin{cases}
                1 & \mbox{if } r_i=0 \\
                (1-\alpha) & \mbox{else}
             \end{cases}
\end{align}
$$

then

$$
\begin{equation}
\mathbf{C}=\vec{s}\vec{z}^T
\end{equation}
$$

So by replacing  ($$\alpha\mathbf{B}^T+(1-\alpha)\vec{s}\vec{1}^T$$) in Eq. $$\eqref{III}$$ with $$\vec{s}\vec{z}^T$$, we'll get:

$$
\begin{align}
    \vec{pr}_{t+1} &= \alpha \mathbf{A}^T\mathbf{D}^{-1}\vec{pr}_{t}+(\vec{s}\vec{z}^T)\vec{pr}_{t}. \tag{IV}\label{IV}
\end{align}
$$

How does this help to improve the calculations? We'll see:

**1. Solving a Linear System**

Similar to before, we can solve Eq. $$\eqref{IV}$$ and get:

$$
\begin{equation}
    \vec{pr}=(I-\alpha \mathbf{A}^T\mathbf{D}^{-1})^{-1}(\vec{s}\vec{z}^T)\vec{pr}.
\end{equation}
$$

Being able to re-parenthesize, $$\vec{z}^T\vec{p}$$ is just a number, so we can ignore it and renormalize $$\vec{pr}$$ at the end, and solve:

$$
\begin{equation}
    \vec{pr}=(I-\alpha \mathbf{A}^T\mathbf{D}^{-1})^{-1}\vec{s}.
\end{equation}
$$

We almost have the same linear equation system that we had before, except for one big improvement, we replaced the less-sparse $$\mathbf{\bar{A}}$$ with $$\mathbf{A}$$.

**2. Power Method**

We can apply one last smart modification to Eq. $$\eqref{IV}$$: if we change the parenthesizing of the last multiplication ([remember the famous dynamic programming algorithm](https://en.wikipedia.org/wiki/Matrix_chain_multiplication)?), and also define $$\mathbf{W}=\alpha\mathbf{A}^T\mathbf{D}^{-1}$$, we will have:

$$
\begin{equation}
\vec{pr}_{t+1} = \mathbf{W}\vec{pr}_{t}+
                \vec{s}(\vec{z}^T\vec{pr}_{t})
\end{equation}
$$

Therefore, the complexity decreased to $$O(n)$$, and the whole recursion will be $$O(n)\times \#iterations$$. The rate of convergence is another thing, which we ignore here, and depends on the value of the second eigenvalue ($$\lambda_2$$) of the modified transition matrix ($$\mathbf{T}$$), which is defined as:
$$
\begin{equation}
\mathbf{T}=\alpha\mathbf{A}^T\mathbf{D}^{-1}+\vec{s}\vec{z}^T
\end{equation}
$$


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

import numpy as np
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
    # In Moler's algorithm, $$A_{ij}$$ represents the existences of an edge
    # from node $$j$$ to $$i$$, while we have assumed the opposite!
    if reverse:
        A = A.T

    n, _ = A.shape
    r = np.asarray(A.sum(axis=1)).reshape(-1)

    k = r.nonzero()[0]

    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = np.ones(n)
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
    # In Moler's algorithm, $$G_{ij}$$ represents the existences of an edge
    # from node $$j$$ to $$i$$, while we have assumed the opposite!
    if reverse:
        A = A.T

    n, _ = A.shape
    r = np.asarray(A.sum(axis=1)).reshape(-1)

    k = r.nonzero()[0]

    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = np.ones(n)
    personalize = personalize.reshape(n, 1)
    s = (personalize / personalize.sum()) * n

    z_T = (((1 - p) * (r != 0) + (r == 0)) / n)[np.newaxis, :]
    W = p * A.T @ D_1

    x = s / n
    oldx = np.zeros((n, 1))

    iteration = 0

    while np.linalg.norm(x - oldx) > tol:
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
import numpy as np
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
        edges1 = np.array([[0, 1],
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
        self.personalize1 = np.array([0.6005, 0.1221, 0.2542, 0.4778, 0.4275])
        self.G1 = sparse.csr_matrix(
            (weights1, (edges1[:, 0], edges1[:, 1])), shape=(n1, n1))
        self.pr1 = np.array([0.1592, 0.2114, 0.3085, 0.1, 0.2208])

        # ---G2---
        n2 = 10
        edges2 = np.array([[2, 4],
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
        self.personalize2 = np.array([0.8887, 0.6491, 0.7843, 0.7103, 0.7428,
                                      0.6632, 0.7351, 0.3006, 0.8722, 0.1652])
        self.p2 = 0.92
        self.pr2 = np.array([0.0234, 0.0255, 0.0629, 0.0196, 0.3303,
                             0.3436, 0.0194, 0.0079, 0.023, 0.1445])

        # ---G3---
        n3 = 5
        edges3 = np.array([[2, 4]])
        weights3 = [0.5441]
        self.G3 = sparse.csr_matrix(
            (weights3, (edges3[:, 0], edges3[:, 1])), shape=(n3, n3))

        self.personalize3 = np.array([0.0884, 0.2797, 0.3093, 0.5533, 0.985])
        self.p3 = 0.81
        self.pr3 = np.array([0.0358, 0.1134, 0.1254, 0.2244, 0.501])

        # ---G4---
        n4 = 5
        edges4_rows = []
        edges4_cols = []
        weights4 = []
        self.G4 = sparse.csr_matrix(
            (weights4, (edges4_rows, edges4_cols)), shape=(n4, n4))

        self.personalize4 = np.array([0.2534, 0.8945, 0.9562, 0.056, 0.9439])
        self.p4 = 0.70
        self.pr4 = np.array([0.0816, 0.2882, 0.3081, 0.018, 0.3041])

        # ---G5---
        n5 = 0
        edges5_rows = []
        edges5_cols = []
        weights5 = []
        self.G5 = sparse.csr_matrix(
            (weights5, (edges5_rows, edges5_cols)), shape=(n5, n5))

        self.personalize5 = np.array([])
        self.p5 = 0.70
        self.pr5 = np.array([])

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

    ..........
    ----------------------------------------------------------------------
    Ran 10 tests in 0.020s
    
    OK


# Benchmarking

To avoid the clutter, we only visualize the fastest method from each implementation, that is:

- `networkx.pagerank`
- Latest implementation of  `iGraph.personalized_pagerank` (PRPACK)
- Our `pagerank_power`



```python
''' Calcualate PageRank on several random graphs.
'''
import numpy as np
import scipy as sp
import pandas as pd
import timeit
import os
import sys
import random
import igraph
import networkx as nx
sys.path.insert(0, '..')
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power

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
    nxG = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph())

    iG = igraph.Graph(list(nxG.edges()), directed=True)
    iG.es['weight'] = A.data

    personalize_vector = np.random.random(G_size)
    personalize_dict = dict(enumerate(personalize_vector.reshape(-1)))
    return A, nxG, iG, personalize_vector, personalize_dict


n = 5
number_of_graphs = 50

node_size_vector = np.zeros(number_of_graphs)
edge_size_vector = np.zeros(number_of_graphs)
netx_pagerank_times = np.zeros(number_of_graphs)
ig_pagerank_times = np.zeros(number_of_graphs)
pagerank_times = np.zeros(number_of_graphs)
pagerank_times_power = np.zeros(number_of_graphs)

damping_factor = 0.85
tol = 1e-3


for i in range(number_of_graphs):
    A, nxG, iG, personalize_vector, personalize_dict = get_random_graph()
    node_size_vector[i] = A.shape[0]
    edge_size_vector[i] = A.count_nonzero()
    print ("Graph %d: Nodes: %d, Edges: %d ..." %(i, node_size_vector[i], edge_size_vector[i]))
    sys.stdout.flush()

    netx_pagerank_times[i] = timeit.timeit(
        lambda: nx.pagerank(nxG, alpha=damping_factor, tol=tol),
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

netx_pagerank_times_sorted = netx_pagerank_times[argsort]

ig_pagerank_times_sorted = ig_pagerank_times[argsort]

pagerank_times_sorted = pagerank_times[argsort]
pagerank_times_power_sorted = pagerank_times_power[argsort]

comparison_table = pd.DataFrame(list(zip(node_size_vector_sorted,
                                         edge_size_vector_sorted,
                                         netx_pagerank_times_sorted,
                                         ig_pagerank_times_sorted,
                                         pagerank_times_sorted,
                                         pagerank_times_power_sorted)),
                                columns=['Nodes', 'Edges',
                                         'NetX',
                                         'iGraph',
                                         '(fast) pagerank',
                                         '(fast) pagerank_power']).\
                    astype({'Nodes': 'int32', 'Edges': 'int32'})
comparison_table.to_csv('pagerank_methods_comparison.csv')
print("Done")
```

    Graph 0: Nodes: 1614, Edges: 892921 ...
    Graph 1: Nodes: 1291, Edges: 397622 ...
    Graph 2: Nodes: 838, Edges: 132426 ...
    Graph 3: Nodes: 673, Edges: 199501 ...
    Graph 4: Nodes: 429, Edges: 27151 ...
    Graph 5: Nodes: 1454, Edges: 825369 ...
    Graph 6: Nodes: 1635, Edges: 579105 ...
    Graph 7: Nodes: 141, Edges: 3475 ...
    Graph 8: Nodes: 706, Edges: 206913 ...
    Graph 9: Nodes: 751, Edges: 137461 ...
    Graph 10: Nodes: 1641, Edges: 1195768 ...
    Graph 11: Nodes: 1413, Edges: 842877 ...
    Graph 12: Nodes: 1042, Edges: 419973 ...
    Graph 13: Nodes: 1681, Edges: 282626 ...
    Graph 14: Nodes: 1089, Edges: 584362 ...
    Graph 15: Nodes: 1260, Edges: 498237 ...
    Graph 16: Nodes: 1736, Edges: 1412119 ...
    Graph 17: Nodes: 1406, Edges: 310850 ...
    Graph 18: Nodes: 240, Edges: 17255 ...
    Graph 19: Nodes: 1176, Edges: 368369 ...
    Graph 20: Nodes: 1083, Edges: 420160 ...
    Graph 21: Nodes: 1933, Edges: 956007 ...
    Graph 22: Nodes: 841, Edges: 145601 ...
    Graph 23: Nodes: 905, Edges: 127398 ...
    Graph 24: Nodes: 858, Edges: 247544 ...
    Graph 25: Nodes: 1989, Edges: 1581139 ...
    Graph 26: Nodes: 869, Edges: 228061 ...
    Graph 27: Nodes: 773, Edges: 164815 ...
    Graph 28: Nodes: 607, Edges: 156039 ...
    Graph 29: Nodes: 1840, Edges: 855672 ...
    Graph 30: Nodes: 349, Edges: 49752 ...
    Graph 31: Nodes: 1722, Edges: 391689 ...
    Graph 32: Nodes: 615, Edges: 170985 ...
    Graph 33: Nodes: 181, Edges: 13687 ...
    Graph 34: Nodes: 1060, Edges: 510199 ...
    Graph 35: Nodes: 956, Edges: 372646 ...
    Graph 36: Nodes: 1784, Edges: 628763 ...
    Graph 37: Nodes: 338, Edges: 32353 ...
    Graph 38: Nodes: 664, Edges: 120154 ...
    Graph 39: Nodes: 888, Edges: 179415 ...
    Graph 40: Nodes: 606, Edges: 173639 ...
    Graph 41: Nodes: 168, Edges: 8709 ...
    Graph 42: Nodes: 159, Edges: 11571 ...
    Graph 43: Nodes: 369, Edges: 67840 ...
    Graph 44: Nodes: 1572, Edges: 672553 ...
    Graph 45: Nodes: 593, Edges: 50826 ...
    Graph 46: Nodes: 603, Edges: 130738 ...
    Graph 47: Nodes: 924, Edges: 152257 ...
    Graph 48: Nodes: 923, Edges: 232844 ...
    Graph 49: Nodes: 532, Edges: 75687 ...
    Done


# Plotting


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

comparison_table = pd.read_csv('pagerank_methods_comparison.csv', index_col=0)
display(comparison_table)

plt.ioff()
fig=plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')


plt.plot(comparison_table['Edges'], comparison_table['NetX'],
         'o-', ms=8, lw=2, alpha=0.7, color='cyan',
         label='networkx.PageRank')

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
      <th>NetX</th>
      <th>iGraph</th>
      <th>(fast) pagerank</th>
      <th>(fast) pagerank_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>141</td>
      <td>3475</td>
      <td>0.008167</td>
      <td>0.000539</td>
      <td>0.001923</td>
      <td>0.000849</td>
    </tr>
    <tr>
      <th>1</th>
      <td>168</td>
      <td>8709</td>
      <td>0.020619</td>
      <td>0.001263</td>
      <td>0.002540</td>
      <td>0.000976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>159</td>
      <td>11571</td>
      <td>0.024342</td>
      <td>0.001638</td>
      <td>0.002263</td>
      <td>0.000924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>181</td>
      <td>13687</td>
      <td>0.026383</td>
      <td>0.001721</td>
      <td>0.002278</td>
      <td>0.000875</td>
    </tr>
    <tr>
      <th>4</th>
      <td>240</td>
      <td>17255</td>
      <td>0.031868</td>
      <td>0.002163</td>
      <td>0.006060</td>
      <td>0.001765</td>
    </tr>
    <tr>
      <th>5</th>
      <td>429</td>
      <td>27151</td>
      <td>0.046971</td>
      <td>0.003327</td>
      <td>0.018867</td>
      <td>0.002158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>338</td>
      <td>32353</td>
      <td>0.062241</td>
      <td>0.004158</td>
      <td>0.012208</td>
      <td>0.002291</td>
    </tr>
    <tr>
      <th>7</th>
      <td>349</td>
      <td>49752</td>
      <td>0.091428</td>
      <td>0.006546</td>
      <td>0.020979</td>
      <td>0.002568</td>
    </tr>
    <tr>
      <th>8</th>
      <td>593</td>
      <td>50826</td>
      <td>0.094839</td>
      <td>0.004848</td>
      <td>0.053456</td>
      <td>0.003128</td>
    </tr>
    <tr>
      <th>9</th>
      <td>369</td>
      <td>67840</td>
      <td>0.129616</td>
      <td>0.009707</td>
      <td>0.015578</td>
      <td>0.003154</td>
    </tr>
    <tr>
      <th>10</th>
      <td>532</td>
      <td>75687</td>
      <td>0.153020</td>
      <td>0.007637</td>
      <td>0.036882</td>
      <td>0.003722</td>
    </tr>
    <tr>
      <th>11</th>
      <td>664</td>
      <td>120154</td>
      <td>0.230722</td>
      <td>0.012611</td>
      <td>0.064293</td>
      <td>0.004689</td>
    </tr>
    <tr>
      <th>12</th>
      <td>905</td>
      <td>127398</td>
      <td>0.243970</td>
      <td>0.015039</td>
      <td>0.184043</td>
      <td>0.005806</td>
    </tr>
    <tr>
      <th>13</th>
      <td>603</td>
      <td>130738</td>
      <td>0.264537</td>
      <td>0.013170</td>
      <td>0.045426</td>
      <td>0.004366</td>
    </tr>
    <tr>
      <th>14</th>
      <td>838</td>
      <td>132426</td>
      <td>0.248668</td>
      <td>0.020659</td>
      <td>0.163919</td>
      <td>0.004816</td>
    </tr>
    <tr>
      <th>15</th>
      <td>751</td>
      <td>137461</td>
      <td>0.267247</td>
      <td>0.013927</td>
      <td>0.076969</td>
      <td>0.004700</td>
    </tr>
    <tr>
      <th>16</th>
      <td>841</td>
      <td>145601</td>
      <td>0.275542</td>
      <td>0.013754</td>
      <td>0.099576</td>
      <td>0.004973</td>
    </tr>
    <tr>
      <th>17</th>
      <td>924</td>
      <td>152257</td>
      <td>0.296603</td>
      <td>0.015513</td>
      <td>0.137361</td>
      <td>0.005365</td>
    </tr>
    <tr>
      <th>18</th>
      <td>607</td>
      <td>156039</td>
      <td>0.295475</td>
      <td>0.014830</td>
      <td>0.053392</td>
      <td>0.005314</td>
    </tr>
    <tr>
      <th>19</th>
      <td>773</td>
      <td>164815</td>
      <td>0.320011</td>
      <td>0.014407</td>
      <td>0.078800</td>
      <td>0.005080</td>
    </tr>
    <tr>
      <th>20</th>
      <td>615</td>
      <td>170985</td>
      <td>0.324415</td>
      <td>0.014716</td>
      <td>0.048411</td>
      <td>0.005089</td>
    </tr>
    <tr>
      <th>21</th>
      <td>606</td>
      <td>173639</td>
      <td>0.329282</td>
      <td>0.017196</td>
      <td>0.047157</td>
      <td>0.005871</td>
    </tr>
    <tr>
      <th>22</th>
      <td>888</td>
      <td>179415</td>
      <td>0.349202</td>
      <td>0.017302</td>
      <td>0.122838</td>
      <td>0.006556</td>
    </tr>
    <tr>
      <th>23</th>
      <td>673</td>
      <td>199501</td>
      <td>0.355692</td>
      <td>0.019995</td>
      <td>0.075585</td>
      <td>0.005577</td>
    </tr>
    <tr>
      <th>24</th>
      <td>706</td>
      <td>206913</td>
      <td>0.405331</td>
      <td>0.020133</td>
      <td>0.068514</td>
      <td>0.005883</td>
    </tr>
    <tr>
      <th>25</th>
      <td>869</td>
      <td>228061</td>
      <td>0.466303</td>
      <td>0.020160</td>
      <td>0.161987</td>
      <td>0.008556</td>
    </tr>
    <tr>
      <th>26</th>
      <td>923</td>
      <td>232844</td>
      <td>0.464954</td>
      <td>0.025322</td>
      <td>0.170835</td>
      <td>0.008981</td>
    </tr>
    <tr>
      <th>27</th>
      <td>858</td>
      <td>247544</td>
      <td>0.498882</td>
      <td>0.028198</td>
      <td>0.130814</td>
      <td>0.007992</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1681</td>
      <td>282626</td>
      <td>0.530677</td>
      <td>0.027410</td>
      <td>0.695906</td>
      <td>0.008940</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1406</td>
      <td>310850</td>
      <td>0.550259</td>
      <td>0.028906</td>
      <td>0.428251</td>
      <td>0.009485</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1176</td>
      <td>368369</td>
      <td>0.715257</td>
      <td>0.044657</td>
      <td>0.396467</td>
      <td>0.017334</td>
    </tr>
    <tr>
      <th>31</th>
      <td>956</td>
      <td>372646</td>
      <td>0.737979</td>
      <td>0.031602</td>
      <td>0.136079</td>
      <td>0.009442</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1722</td>
      <td>391689</td>
      <td>0.763951</td>
      <td>0.047625</td>
      <td>0.849482</td>
      <td>0.015669</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1291</td>
      <td>397622</td>
      <td>0.704370</td>
      <td>0.034768</td>
      <td>0.311621</td>
      <td>0.009304</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1042</td>
      <td>419973</td>
      <td>0.768082</td>
      <td>0.043818</td>
      <td>0.184774</td>
      <td>0.011507</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1083</td>
      <td>420160</td>
      <td>1.006566</td>
      <td>0.158505</td>
      <td>1.352094</td>
      <td>0.011576</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1260</td>
      <td>498237</td>
      <td>1.011772</td>
      <td>0.059739</td>
      <td>0.368076</td>
      <td>0.011779</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1060</td>
      <td>510199</td>
      <td>1.140068</td>
      <td>0.059229</td>
      <td>0.237280</td>
      <td>0.019025</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1635</td>
      <td>579105</td>
      <td>1.137819</td>
      <td>0.061414</td>
      <td>0.767978</td>
      <td>0.013246</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1089</td>
      <td>584362</td>
      <td>1.037787</td>
      <td>0.074236</td>
      <td>0.254159</td>
      <td>0.012441</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1784</td>
      <td>628763</td>
      <td>1.427946</td>
      <td>0.058649</td>
      <td>0.877097</td>
      <td>0.015091</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1572</td>
      <td>672553</td>
      <td>1.295139</td>
      <td>0.056538</td>
      <td>0.638620</td>
      <td>0.015338</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1454</td>
      <td>825369</td>
      <td>1.507927</td>
      <td>0.107227</td>
      <td>0.405008</td>
      <td>0.014615</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1413</td>
      <td>842877</td>
      <td>1.624581</td>
      <td>0.111469</td>
      <td>0.444221</td>
      <td>0.015862</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1840</td>
      <td>855672</td>
      <td>1.623843</td>
      <td>0.075287</td>
      <td>0.838486</td>
      <td>0.016817</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1614</td>
      <td>892921</td>
      <td>1.710065</td>
      <td>0.133828</td>
      <td>0.627106</td>
      <td>0.015474</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1933</td>
      <td>956007</td>
      <td>2.507147</td>
      <td>0.119857</td>
      <td>1.135223</td>
      <td>0.016590</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1641</td>
      <td>1195768</td>
      <td>2.196352</td>
      <td>0.172022</td>
      <td>0.535405</td>
      <td>0.019178</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1736</td>
      <td>1412119</td>
      <td>2.638824</td>
      <td>0.192373</td>
      <td>0.674815</td>
      <td>0.021137</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1989</td>
      <td>1581139</td>
      <td>3.274551</td>
      <td>0.256537</td>
      <td>1.405950</td>
      <td>0.020392</td>
    </tr>
  </tbody>
</table>
</div>



![png](/assets/Fast-PageRank_9_1.png)
