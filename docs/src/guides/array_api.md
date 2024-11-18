```@meta
CurrentModule = Finch
```

# High-Level Array API

Finch tensors also support many of the basic array operations one might expect,
including indexing, slicing, and elementwise maps, broadcast, and reduce.
For example:

```jldoctest example1; setup = :(using Finch)
julia> A = fsparse([1, 1, 2, 3], [2, 4, 5, 6], [1.0, 2.0, 3.0])
3×6-Tensor
└─ SparseCOO{2} (0.0) [:,1:6]
   ├─ [1, 2]: 1.0
   ├─ [1, 4]: 2.0
   └─ [2, 5]: 3.0

julia> A + 0
3×6-Tensor
└─ Dense [:,1:6]
   ├─ [:, 1]: Dense [1:3]
   │  ├─ [1]: 0.0
   │  ├─ [2]: 0.0
   │  └─ [3]: 0.0
   ├─ [:, 2]: Dense [1:3]
   │  ├─ [1]: 1.0
   │  ├─ [2]: 0.0
   │  └─ [3]: 0.0
   ├─ ⋮
   ├─ [:, 5]: Dense [1:3]
   │  ├─ [1]: 0.0
   │  ├─ [2]: 3.0
   │  └─ [3]: 0.0
   └─ [:, 6]: Dense [1:3]
      ├─ [1]: 0.0
      ├─ [2]: 0.0
      └─ [3]: 0.0

julia> A + 1
3×6-Tensor
└─ Dense [:,1:6]
   ├─ [:, 1]: Dense [1:3]
   │  ├─ [1]: 1.0
   │  ├─ [2]: 1.0
   │  └─ [3]: 1.0
   ├─ [:, 2]: Dense [1:3]
   │  ├─ [1]: 2.0
   │  ├─ [2]: 1.0
   │  └─ [3]: 1.0
   ├─ ⋮
   ├─ [:, 5]: Dense [1:3]
   │  ├─ [1]: 1.0
   │  ├─ [2]: 4.0
   │  └─ [3]: 1.0
   └─ [:, 6]: Dense [1:3]
      ├─ [1]: 1.0
      ├─ [2]: 1.0
      └─ [3]: 1.0

julia> B = A .* 2
3×6-Tensor
└─ SparseDict (0.0) [:,1:6]
   ├─ [:, 2]: SparseDict (0.0) [1:3]
   │  └─ [1]: 2.0
   ├─ [:, 4]: SparseDict (0.0) [1:3]
   │  └─ [1]: 4.0
   └─ [:, 5]: SparseDict (0.0) [1:3]
      └─ [2]: 6.0

julia> B[1:2, 1:2]
2×2-Tensor
└─ SparseDict (0.0) [:,1:2]
   └─ [:, 2]: SparseDict (0.0) [1:2]
      └─ [1]: 2.0

julia> map(x -> x^2, B)
3×6-Tensor
└─ SparseDict (0.0) [:,1:6]
   ├─ [:, 2]: SparseDict (0.0) [1:3]
   │  └─ [1]: 4.0
   ├─ [:, 4]: SparseDict (0.0) [1:3]
   │  └─ [1]: 16.0
   └─ [:, 5]: SparseDict (0.0) [1:3]
      └─ [2]: 36.0
```

# Array Fusion

Finch supports array fusion, which allows you to compose multiple array operations
into a single kernel. This can be a significant performance optimization, as it
allows the compiler to optimize the entire operation at once. The two functions
the user needs to know about are `lazy` and `compute`. You can use `lazy` to
mark an array as an input to a fused operation, and call `compute` to execute
the entire operation at once. For example:

```jldoctest example1
julia> C = lazy(A);

julia> D = lazy(B);

julia> E = (C .+ D)/2;

julia> compute(E)
3×6-Tensor
└─ SparseDict (0.0) [:,1:6]
   ├─ [:, 2]: SparseDict (0.0) [1:3]
   │  └─ [1]: 1.5
   ├─ [:, 4]: SparseDict (0.0) [1:3]
   │  └─ [1]: 3.0
   └─ [:, 5]: SparseDict (0.0) [1:3]
      └─ [2]: 4.5

```

In the above example, `E` is a fused operation that adds `C` and `D` together
and then divides the result by 2. The `compute` function examines the entire
operation and decides how to execute it in the most efficient way possible.
In this case, it would likely generate a single kernel that adds the elements of `A` and `B`
together and divides each result by 2, without materializing an intermediate.

```@docs
lazy
compute
```

## The Galley Optimizer

Galley is a cost-based optimizer for Finch's lazy evaluation interface based on techniques from database 
query optimization. To use Galley, you just add the parameter `ctx=galley_optimizer()` to the `compute` 
function. While the default optimizer (`ctx=default_scheduler()`) makes decisions entirely based on
the types of the inputs, Galley gathers statistics on their sparsity to make cost-based based optimization
decisions.

Consider the following set of small examples:

```
   N = 300
   A = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, .5)))
   B = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, .5)))
   C = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, .01)))

   println("Galley: A * B * C")
   empty!(Finch.codes)
   @btime begin 
      compute($A * $B * $C, ctx=galley_scheduler())
   end

   println("Galley: C * B * A")
   empty!(Finch.codes)
   @btime begin 
      compute($C * $B * $A, ctx=galley_scheduler())
   end

   println("Galley: sum(C * B * A)")
   empty!(Finch.codes)
   @btime begin 
      compute(sum($C * $B * $A), ctx=galley_scheduler())
   end

   println("Finch: A * B * C")
   empty!(Finch.codes)
   @btime begin 
      compute($A * $B * $C, ctx=Finch.default_scheduler())
   end

   println("Finch: C * B * A")
   empty!(Finch.codes)
   @btime begin 
      compute($C * $B * $A, ctx=Finch.default_scheduler())
   end

   println("Finch: sum(C * B * A)")
   empty!(Finch.codes)
   @btime begin 
      compute(sum($C * $B * $A), ctx=Finch.default_scheduler())
   end
```

By taking advantage of the fact that C is highly sparse, Galley can better structure the computation. In the matrix chain multiplication,
it always starts with the C,B matmul before multiplying with A. In the summation, it takes advantage of distributivity to pushing the reduction
down to the inputs. It first sums over A and C, then multiplies those vectors with B.

# Einsum

Finch also supports a highly general `@einsum` macro which supports any reduction over any simple pointwise array expression.

```@docs
@einsum
```