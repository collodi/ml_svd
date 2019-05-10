ml_svd
===

Machine learning using singular-value decomposition


Problem
---

We need a layer that can handle large input with a relatively small number of parameters.

*Convolutional networks?*
True, but they don't work too well on discrete data.


*So, how well does this perform on discrete data?*
I have no idea yet. More experiments will be posted.


*Where is SVD?*
Trust me, or just look at the next section.


Reasoning behind the design
---

So, given some input matrix `X`, `U, E, V = svd(X)`.

Now, an SVDLayer has 2 fully-connected layers inside.
They do not have biases.
So, they are simply 2 matrices `A` and `B`.

Intuitively, we can think of columns of `U` and `V` as some information embedded in the input matrix `X`.
Assume `X` has size `h` by `w`.
Then, each column of `U` and `V` have lengths `h` and `w`, respectively.

An `SVDLayer` is defined with the input size `(h, w)` and the output size `(h0, w0)`.
Now, `A` transforms each column of `U` into a vector of length `h0`.
Similarly, `B` transforms each column of `V` into a vector of length `w0`.

Let's say the output of `SVDLayer` is `Y`.
Equivalently, `Y = SVDLayer(X)`.

Let `U0 = A U` and `V0 = B V`.

Here comes a magic trick that removes SVD from `SVDLayer`.

```
Y = U0 E V0^
  = (A U) E (B V)^
  = A U E V^ B^
  = A   X    B^
```

Tada?
