The trend we're seeing is that the first reshape is fast, the einsum is a little faster, but the last reshape is about as fast as the einsum. This is probably where the 2x slowdown comes from.

(1000x10x1000) x (1000, 1000)

With Reshape:
Reshape 1 : 0.064485228 
Einsum    : 0.644003844 
Reshape 2 : 0.612126186

Without Reshape:
Einsum    : 0.7851503

(500, 500, 500) x (500, 500)

With Reshape:
Reshape 1 : 0.173063197 
Einsum    : 1.058090215 
Reshape 2 : 1.268889115

Without Reshape:
Einsum    : 1.388311393


TODO:

- nnz vs runtime of einsum
- runtime without last reshape
