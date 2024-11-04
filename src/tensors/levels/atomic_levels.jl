
"""
    AtomicLevel{Val, Lvl}()

Atomic Level Protects the level directly below it with atomics

Each position in the level below the atomic level is protected by an atomic.
```jldoctest
julia> Tensor(Dense(Atomic(Element(0.0))), [1, 2, 3])
Dense [1:3]
├─[1]: Atomic -> 1.0
├─[2]: Atomic -> 2.0
├─[3]: Atomic -> 3.0
```
"""

struct AtomicLevel{AVal, Lvl} <: AbstractLevel
    lvl::Lvl
    locks::AVal
end
const Atomic = AtomicLevel


AtomicLevel(lvl) = AtomicLevel(lvl, Base.Threads.SpinLock[])
#AtomicLevel(lvl::Lvl, locks::AVal) where {Lvl, AVal} =
#    AtomicLevel{AVal, Lvl}(lvl, locks)
Base.summary(::AtomicLevel{AVal, Lvl}) where {Lvl, AVal} = "AtomicLevel($(AVal), $(Lvl))"

similar_level(lvl::Atomic{AVal, Lvl}, fill_value, eltype::Type, dims...) where {Lvl, AVal} =
    AtomicLevel(similar_level(lvl.lvl, fill_value, eltype, dims...))

postype(::Type{<:AtomicLevel{AVal, Lvl}}) where {Lvl, AVal} = postype(Lvl)

function moveto(lvl::AtomicLevel, device)
    lvl_2 = moveto(lvl.lvl, device)
    locks_2 = moveto(lvl.locks, device)
    return AtomicLevel(lvl_2, locks_2)
end

pattern!(lvl::AtomicLevel) = AtomicLevel(pattern!(lvl.lvl), lvl.locks)
set_fill_value!(lvl::AtomicLevel, init) = AtomicLevel(set_fill_value!(lvl.lvl, init), lvl.locks)
# TODO: FIXME: Need toa dopt the number of dims
Base.resize!(lvl::AtomicLevel, dims...) = AtomicLevel(resize!(lvl.lvl, dims...), lvl.locks)


function Base.show(io::IO, lvl::AtomicLevel{AVal, Lvl}) where {AVal, Lvl}
    print(io, "Atomic(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(IOContext(io), lvl.lvl)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>AVal), lvl.locks)
    end
    print(io, ")")
end

labelled_show(io::IO, ::SubFiber{<:AtomicLevel}) =
    print(io, "Atomic -> ")

function labelled_children(fbr::SubFiber{<:AtomicLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    [LabelledTree(SubFiber(lvl.lvl, pos))]
end


@inline level_ndims(::Type{<:AtomicLevel{AVal, Lvl}}) where {AVal, Lvl} = level_ndims(Lvl)
@inline level_size(lvl::AtomicLevel{AVal, Lvl}) where {AVal, Lvl} = level_size(lvl.lvl)
@inline level_axes(lvl::AtomicLevel{AVal, Lvl}) where {AVal, Lvl} = level_axes(lvl.lvl)
@inline level_eltype(::Type{AtomicLevel{AVal, Lvl}}) where {AVal, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:AtomicLevel{AVal, Lvl}}) where {AVal, Lvl} = level_fill_value(Lvl)

# FIXME: These.
(fbr::Tensor{<:AtomicLevel})() = SubFiber(fbr.lvl, 1)()
(fbr::SubFiber{<:AtomicLevel})() = fbr #TODO this is not consistent somehow
function (fbr::SubFiber{<:AtomicLevel})(idxs...)
    return Tensor(fbr.lvl.lvl)(idxs...)
end

countstored_level(lvl::AtomicLevel, pos) = countstored_level(lvl.lvl, pos)

mutable struct VirtualAtomicLevel <: AbstractVirtualLevel
    lvl # the level below us.
    ex
    locks
    Tv
    Val
    AVal
    Lvl
end
postype(lvl:: AtomicLevel) = postype(lvl.lvl)

postype(lvl:: VirtualAtomicLevel) = postype(lvl.lvl)

is_level_injective(ctx, lvl::VirtualAtomicLevel) = [is_level_injective(ctx, lvl.lvl)...]
function is_level_concurrent(ctx, lvl::VirtualAtomicLevel)
    (below, c) = is_level_concurrent(ctx, lvl.lvl)
    return (below, c)
end
function is_level_atomic(ctx, lvl::VirtualAtomicLevel)
    (below, _) = is_level_atomic(ctx, lvl.lvl)
    return (below, true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualAtomicLevel, ::DefaultStyle)
    quote
        $AtomicLevel{$(lvl.AVal), $(lvl.Lvl)}($(ctx(lvl.lvl)), $(lvl.locks))
    end
end

function virtualize(ctx, ex, ::Type{AtomicLevel{AVal, Lvl}}, tag=:lvl) where {AVal, Lvl}
    sym = freshen(ctx, tag)
    atomics = freshen(ctx, tag, :_locks)
    push_preamble!(ctx, quote
            $sym = $ex
            $atomics = $ex.locks
        end)
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    temp = VirtualAtomicLevel(lvl_2, sym, atomics, typeof(level_fill_value(Lvl)), Val, AVal, Lvl)
    temp
end

Base.summary(lvl::VirtualAtomicLevel) = "Atomic($(lvl.Lvl))"
virtual_level_resize!(ctx, lvl::VirtualAtomicLevel, dims...) = (lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...); lvl)
virtual_level_size(ctx, lvl::VirtualAtomicLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_ndims(ctx, lvl::VirtualAtomicLevel) = length(virtual_level_size(ctx, lvl.lvl))
virtual_level_eltype(lvl::VirtualAtomicLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualAtomicLevel) = virtual_level_fill_value(lvl.lvl)

function declare_level!(ctx, lvl::VirtualAtomicLevel, pos, init)
    lvl.lvl = declare_level!(ctx, lvl.lvl, pos, init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualAtomicLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    idx = freshen(ctx, :idx)
    lockVal = freshen(ctx, :lock)
    push_preamble!(ctx, quote
              Finch.resize_if_smaller!($(lvl.locks), $(ctx(pos_stop)))
              @inbounds for $idx = $(ctx(pos_start)):$(ctx(pos_stop))
                $(lvl.locks)[$idx] = Finch.make_lock(eltype($(lvl.AVal)))
              end
          end)
    assemble_level!(ctx, lvl.lvl, pos_start, pos_stop)
end

supports_reassembly(lvl::VirtualAtomicLevel) = supports_reassembly(lvl.lvl)
function reassemble_level!(ctx, lvl::VirtualAtomicLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    idx = freshen(ctx, :idx)
    lockVal = freshen(ctx, :lock)
    push_preamble!(ctx, quote
              Finch.resize_if_smaller!($lvl.locks, $(ctx(pos_stop)))
              @inbounds for $idx = $(ctx(pos_start)):$(ctx(pos_stop))
                $lvl.locks[$idx] = Finch.make_lock(eltype($(lvl.AVal)))
              end
          end)
    reassemble_level!(ctx, lvl.lvl, pos_start, pos_stop)
    lvl
end

function freeze_level!(ctx, lvl::VirtualAtomicLevel, pos)
    idx = freshen(ctx, :idx)
    push_preamble!(ctx, quote
        resize!($(lvl.locks), $(ctx(pos)))
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, pos)
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualAtomicLevel, pos)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, pos)
    return lvl
end

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualAtomicLevel, arch)
    #Add for seperation level too.
    atomics = freshen(ctx, :locksArray)

    push_preamble!(ctx, quote
        $atomics = $(lvl.locks)
        $(lvl.locks) = $moveto($(lvl.locks), $(ctx(arch)))
    end)
    push_epilogue!(ctx, quote
        $(lvl.locks) = $atomics
    end)
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

function unwrap_outer(ctx, fbr::VirtualSubFiber{VirtualAtomicLevel}, mode::Reader, protos)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    # lvlp = freshen(ctx, lvl.ex, :_lvl)
    # sym = freshen(ctx, lvl.ex, :_after_atomic_lvl)
    return body = Thunk(
        body = (ctx) -> begin
            unwrap_outer(ctx, VirtualSubFiber(lvl.lvl, pos), mode, protos)
        end,
    )
end

function unwrap_outer(ctx, fbr::VirtualSubFiber{VirtualAtomicLevel}, mode::Updater, protos)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    sym = freshen(ctx, lvl.ex, :after_atomic_lvl)
    atomicData = freshen(ctx, lvl.ex, :atomicArraysAcc)
    lockVal = freshen(ctx, lvl.ex, :lockVal)
    dev = lower(ctx, virtual_get_device(ctx.code.task), DefaultStyle())
    return Thunk(
        body =  (ctx) -> begin
        preamble = quote
            $atomicData =  Finch.get_lock($dev, $(lvl.locks), $(ctx(pos)), eltype($(lvl.AVal)))
            $lockVal = Finch.aquire_lock!($dev, $atomicData)
        end
        epilogue = quote
            Finch.release_lock!($dev, $atomicData) end
        push_preamble!(ctx, preamble)
        push_epilogue!(ctx, epilogue)
            lvl_2 = lvl.lvl
            update = unwrap_outer(ctx, VirtualSubFiber(lvl_2, pos), mode, protos)
            return update
        end,

    )
end
function unwrap_outer(ctx, fbr::VirtualHollowSubFiber{VirtualAtomicLevel}, mode::Updater, protos)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    sym = freshen(ctx, lvl.ex, :after_atomic_lvl)
    atomicData = freshen(ctx, lvl.ex, :atomicArrays)
    lockVal = freshen(ctx, lvl.ex, :lockVal)
    dev = lower(ctx, virtual_get_device(ctx.code.task), DefaultStyle())
    return Thunk(

        body =  (ctx) -> begin
        preamble = quote
            $atomicData =  Finch.get_lock($dev, $(lvl.locks), $(ctx(pos)), eltype($(lvl.AVal)))
            $lockVal = Finch.aquire_lock!($dev, $atomicData)
        end
        epilogue = quote
            Finch.release_lock!($dev, $atomicData) end
            push_preamble!(ctx, preamble)
            push_epilogue!(ctx, epilogue)
            lvl_2 = lvl.lvl
            update = unwrap_outer(ctx, VirtualHollowSubFiber(lvl_2, pos, fbr.dirty), mode, protos)
            return update
        end
    )
end