"""
    bfs(edges; [source])

Calculate a breadth-first search on the graph specified by the `edges` adjacency
matrix. Return the node numbering.
"""
function bfs(edges, source=5)
    (n, m) = size(edges)
    # 1 for all non-zero values else 0
    edges = pattern!(edges)

    @assert n == m
    # current frontier
    F = Tensor(SparseByteMap(Pattern()), n)
    # next frontier
    _F = Tensor(SparseByteMap(Pattern()), n)
    # mark source as visited
    @finch F[source] = true
    # count of # of nonzero entries
    F_nnz = 1

    # visited vector
    V = Tensor(Dense(Element(false)), n)
    @finch V[source] = true

    # parent of each value
    P = Tensor(Dense(Element(0)), n)
    @finch P[source] = source

    while F_nnz > 0
        @finch begin
            # reset tmp frontier
            _F .= false
            for j in _, k in _

                # if value in frontier and edge and not visited
                if F[j] && edges[k, j] && !(V[k])
                    _F[k] |= true
                    P[k] << choose(0) >>= j #Only set the parent for this vertex
                end
            end
        end

        # count # of new nodes and update visited
        c = Scalar(0)
        @finch begin
            for k in _
                let _f = _F[k]
                    V[k] |= _f
                    c[] += _f
                end
            end
        end

        # update frontier and nonzeros
        (F, _F) = (_F, F)
        F_nnz = c[]
    end
    return P
end
