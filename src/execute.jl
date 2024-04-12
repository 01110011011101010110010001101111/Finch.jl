abstract type CompileMode end
struct DebugFinch <: CompileMode end
const debugfinch = DebugFinch()
virtualize(ctx, ex, ::Type{DebugFinch}) = DebugFinch()
struct SafeFinch <: CompileMode end
const safefinch = SafeFinch()
virtualize(ctx, ex, ::Type{SafeFinch}) = SafeFinch()
struct FastFinch <: CompileMode end
const fastfinch = FastFinch()
virtualize(ctx, ex, ::Type{FastFinch}) = FastFinch()

issafe(::DebugFinch) = true
issafe(::SafeFinch) = true
issafe(::FastFinch) = false

"""
    instantiate!(ctx, prgm)

A transformation to instantiate readers and updaters before executing an
expression.
"""
function instantiate!(ctx, prgm) 
    prgm = InstantiateTensors(ctx=ctx)(prgm)
    return prgm
end

@kwdef struct InstantiateTensors{Ctx}
    ctx::Ctx
    escape = Set()
end

function (ctx::InstantiateTensors)(node::FinchNode)
    if node.kind === block
        block(map(ctx, node.bodies)...)
    elseif node.kind === define
        push!(ctx.escape, node.lhs)
        define(node.lhs, ctx(node.rhs), ctx(node.body))
    elseif node.kind === declare
        push!(ctx.escape, node.tns)
        node
    elseif node.kind === freeze
        push!(ctx.escape, node.tns)
        node
    elseif node.kind === thaw
        push!(ctx.escape, node.tns)
        node
    elseif (@capture node access(~tns, ~mode, ~idxs...)) && !(getroot(tns) in ctx.escape)
        #@assert get(ctx.ctx.modes, tns, reader) === node.mode.val
        protos = [(mode.val === reader ? defaultread : defaultupdate) for _ in idxs]
        tns_2 = instantiate(tns, ctx.ctx, mode.val, protos)
        access(tns_2, mode, idxs...)
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

execute(ex) = execute(ex, NamedTuple())

@staged function execute(ex, opts)
    contain(JuliaContext()) do ctx
        code = execute_code(:ex, ex; virtualize(ctx, :opts, opts)...)
        quote
            # try
                @inbounds @fastmath begin
                    $(code |> unblock)
                end
            # catch
            #    println("Error executing code:")
            #    println($(QuoteNode(code |> unblock |> pretty |> unquote_literals)))
            #    rethrow()
            #end
        end
    end
end

function execute_code(ex, T; algebra = DefaultAlgebra(), mode = safefinch, ctx = LowerJulia(algebra = algebra, mode=mode))
    code = contain(ctx) do ctx_2
        prgm = nothing
        prgm = virtualize(ctx_2.code, ex, T)
        lower_global(ctx_2, prgm)
    end
end

"""
    lower_global(ctx, prgm)

lower the program `prgm` at global scope in the context `ctx`.
"""
function lower_global(ctx, prgm)
    prgm = enforce_scopes(prgm)
    prgm = evaluate_partial(ctx, prgm)
    code = contain(ctx) do ctx_2
        quote
            $(ctx.needs_return) = true
            $(ctx.result) = nothing
            $(begin
                prgm = wrapperize(ctx_2, prgm)
                prgm = enforce_lifecycles(prgm)
                prgm = dimensionalize!(prgm, ctx_2)
                prgm = concordize(ctx_2, prgm)
                prgm = evaluate_partial(ctx_2, prgm)
                prgm = simplify(ctx_2, prgm) #appears necessary
                prgm = instantiate!(ctx_2, prgm)
                contain(ctx_2) do ctx_3
                    ctx_3(prgm)
                end
            end)
            $(ctx.result)
        end
    end
end

"""
    @finch [options...] prgm

Run a finch program `prgm`. The syntax for a finch program is a set of nested
loops, statements, and branches over pointwise array assignments. For example,
the following program computes the sum of two arrays `A = B + C`:

```julia   
@finch begin
    A .= 0
    for i = _
        A[i] = B[i] + C[i]
    end
    return A
end
```

Finch programs are composed using the following syntax:

 - `arr .= 0`: an array declaration initializing arr to zero.
 - `arr[inds...]`: an array access, the array must be a variable and each index may be another finch expression.
 - `x + y`, `f(x, y)`: function calls, where `x` and `y` are finch expressions.
 - `arr[inds...] = ex`: an array assignment expression, setting `arr[inds]` to the value of `ex`.
 - `arr[inds...] += ex`: an incrementing array expression, adding `ex` to `arr[inds]`. `*, &, |`, are supported.
 - `arr[inds...] <<min>>= ex`: a incrementing array expression with a custom operator, e.g. `<<min>>` is the minimum operator.
 - `for i = _ body end`: a loop over the index `i`, where `_` is computed from array access with `i` in `body`.
 - `if cond body end`: a conditional branch that executes only iterations where `cond` is true.
 - `return (tnss...,)`: at global scope, exit the program and return the tensors `tnss` with their new dimensions. By default, any tensor declared in global scope is returned.

Symbols are used to represent variables, and their values are taken from the environment. Loops introduce
index variables into the scope of their bodies.

Finch uses the types of the arrays and symbolic analysis to discover program
optimizations. If `B` and `C` are sparse array types, the program will only run
over the nonzeros of either. 

Semantically, Finch programs execute every iteration. However, Finch can use
sparsity information to reliably skip iterations when possible.

`options` are optional keyword arguments:

 - `algebra`: the algebra to use for the program. The default is `DefaultAlgebra()`.
 - `mode`: the optimization mode to use for the program. The default is `fastfinch`.
 - `ctx`: the context to use for the program. The default is a `LowerJulia` context with the given options.

See also: [`@finch_code`](@ref)
"""
macro finch(opts_ex...)
    length(opts_ex) >= 1 || throw(ArgumentError("Expected at least one argument to @finch(opts..., ex)"))
    (opts, ex) = (opts_ex[1:end-1], opts_ex[end])
    prgm = FinchNotation.finch_parse_instance(ex)
    prgm = :(
        $(FinchNotation.block_instance)(
            $prgm,
            $(FinchNotation.yieldbind_instance)(
                $(map(FinchNotation.variable_instance, FinchNotation.finch_parse_default_yieldbind(ex))...)
            )
        )
    )
    res = esc(:res)
    thunk = quote
        res = $execute($prgm, (;$(map(esc, opts)...),))
    end
    for tns in something(FinchNotation.finch_parse_yieldbind(ex), FinchNotation.finch_parse_default_yieldbind(ex))
        push!(thunk.args, quote
            $(esc(tns)) = res[$(QuoteNode(tns))]
        end)
    end
    push!(thunk.args, quote
        res
    end)
    thunk
end

"""
@finch_code [options...] prgm

Return the code that would be executed in order to run a finch program `prgm`.

See also: [`@finch`](@ref)
"""
macro finch_code(opts_ex...)
    length(opts_ex) >= 1 || throw(ArgumentError("Expected at least one argument to @finch(opts..., ex)"))
    (opts, ex) = (opts_ex[1:end-1], opts_ex[end])
    prgm = FinchNotation.finch_parse_instance(ex)
    prgm = :(
        $(FinchNotation.block_instance)(
            $prgm,
            $(FinchNotation.yieldbind_instance)(
                $(map(FinchNotation.variable_instance, FinchNotation.finch_parse_default_yieldbind(ex))...)
            )
        )
    )
    return quote
        $execute_code(:ex, typeof($prgm); $(map(esc, opts)...)) |> pretty |> unresolve |> dataflow |> unquote_literals
    end
end

"""
    finch_kernel(fname, args, prgm; options...)

Return a function definition for which can execute a Finch program of
type `prgm`. Here, `fname` is the name of the function and `args` is a
`iterable` of argument name => type pairs.

See also: [`@finch`](@ref)
"""
function finch_kernel(fname, args, prgm; algebra = DefaultAlgebra(), mode = safefinch, ctx = LowerJulia(algebra=algebra, mode=mode))
    maybe_typeof(x) = x isa Type ? x : typeof(x)
    code = contain(ctx) do ctx_2
        foreach(args) do (key, val)
            ctx_2.bindings[variable(key)] = finch_leaf(virtualize(ctx_2.code, key, maybe_typeof(val), key))
        end
        execute_code(:UNREACHABLE, prgm, algebra = algebra, mode = mode, ctx = ctx_2)
    end |> pretty |> unresolve |> dataflow |> unquote_literals
    arg_defs = map(((key, val),) -> :($key::$(maybe_typeof(val))), args)
    striplines(:(function $fname($(arg_defs...))
        @inbounds @fastmath $(striplines(unblock(code)))
    end))
end

"""
    @finch_kernel [options...] fname(args...) = prgm

Return a definition for a function named `fname` which executes `@finch prgm` on
the arguments `args`. `args` should be a list of variables holding
representative argument instances or types.

See also: [`@finch`](@ref)
"""
macro finch_kernel(opts_def...)
    length(opts_def) >= 1 || throw(ArgumentError("expected at least one argument to @finch(opts..., def)"))
    (opts, def) = (opts_def[1:end-1], opts_def[end])
    (@capture def :function(:call(~name, ~args...), ~ex)) ||
    (@capture def :(=)(:call(~name, ~args...), ~ex)) ||
    throw(ArgumentError("unrecognized function definition in @finch_kernel"))
    named_args = map(arg -> :($(QuoteNode(arg)) => $(esc(arg))), args)
    prgm = FinchNotation.finch_parse_instance(ex)
    for arg in args
        prgm = quote    
            let $(esc(arg)) = $(FinchNotation.variable_instance(arg))
                $prgm
            end
        end
    end
    return quote
        $finch_kernel($(QuoteNode(name)), Any[$(named_args...),], typeof($prgm); $(map(esc, opts)...))
    end
end
