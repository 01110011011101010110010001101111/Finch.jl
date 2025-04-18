@testitem "examples" begin
    using SparseArrays
    using LinearAlgebra
    using Graphs, SimpleWeightedGraphs

    include(joinpath(@__DIR__, "../../docs/examples/bfs.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/dfs.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/pagerank.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/shortest_paths.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/spgemm.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/triangle_counting.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/degree_centrality.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/topological_sort.jl"))
    include(joinpath(@__DIR__, "../../docs/examples/floyd_warshall.jl"))

    @testset "pagerank" begin
        size, sparsity = 30, 0.5
        input = sprand(size, size, sparsity)

        graphs_input = Graphs.SimpleDiGraph(transpose(input))
        finch_input = pattern!(Tensor(input))

        expected = Graphs.pagerank(graphs_input, 0.85, 20)
        output = pagerank(finch_input; nsteps=20, damp=0.85)

        tol = 1e-6

        output = copyto!(zeros(size), output)

        @test maximum(abs.(output .- expected)) < tol
    end

    @testset "bfs" begin
        size, sparsity = 50, 0.5
        source = rand(1:size)
        input = sprand(size, size, sparsity)

        graphs_input = Graphs.SimpleDiGraph(transpose(input))
        finch_input = Tensor(Dense(SparseList(Element(0.0))), input)

        expected = Graphs.bfs_parents(graphs_input, source)
        output = bfs(finch_input, source)

        @test output == expected
    end

    @testset "dfs" begin
        size, sparsity = 50, 0.5
        source = rand(1:size)
        input = sprand(size, size, sparsity)

        graphs_input = Graphs.SimpleDiGraph(transpose(input))
        finch_input = Tensor(Dense(SparseList(Element(0.0))), input)

        expected = Graphs.dfs_parents(graphs_input, source)
        output = dfs(finch_input, source)

        @test output == expected
    end

    @testset "degree_centrality" begin
        size, sparsity = 50, 0.5
        source = rand(1:size)
        input = sprand(size, size, sparsity)

        graphs_input = Graphs.SimpleDiGraph(transpose(input))
        finch_input = Tensor(Dense(SparseList(Element(0.0))), input)

        expected = Graphs.degree_centrality(graphs_input)
        output = degree_centrality(finch_input)

        @test output == expected
    end

    @testset "bellmanford" begin
        size, sparsity = 50, 0.5
        source = rand(1:size)
        input = sprand(size, size, sparsity)

        graphs_input = SimpleWeightedDiGraph(transpose(input))
        finch_input = set_fill_value!(Tensor(Dense(SparseList(Element(0.0))), input), Inf)

        expected = Graphs.bellman_ford_shortest_paths(graphs_input, source)
        output = bellmanford(finch_input, source)

        @test output == collect(zip(expected.dists, expected.parents))
    end

    @testset "tricount" begin
        size, sparsity = 1000, 0.5
        input = sprand(size, size, sparsity)
        input = SparseMatrixCSC(Symmetric(input))

        graphs_input = SimpleDiGraph(input)
        finch_input = pattern!(Tensor(Dense(SparseList(Element(0.0))), input))

        expected = sum(Graphs.triangles(graphs_input))
        output = tricount(finch_input) * 6

        @test expected == output
    end

    @testset "spgemm" begin
        m, n, k = (32, 32, 32)
        p = 0.1
        A_ref = sprand(Int, m, k, p)
        B_ref = sprand(Int, k, n, p)
        C_ref = A_ref * B_ref
        A = Tensor(Dense(SparseList(Element(0))), A_ref)
        B = Tensor(Dense(SparseList(Element(0))), B_ref)

        for (key, fn) in [
            (:spgemm_inner, spgemm_inner),
            (:spgemm_gustavson, spgemm_gustavson),
            (:spgemm_outer, spgemm_outer),
        ]
            C = fn(A, B)
            @test C == C_ref
        end
    end

    @testset "topological_sort" begin
        A = Tensor(
            CSCFormat(0),
            [0 0 0 0 1 1; 0 0 0 1 1 0; 0 0 0 0 0 1; 0 0 1 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0],
        )

        @test topological_sort(A) == [5, 6, 1, 3, 4, 2]
    end

    @testset "floyd_warshall" begin
        # https://cses.fi/problemset/task/1672
        matrix = Tensor(
            CSCFormat(Inf), [0 5 9 Inf; Inf 0 3 Inf; Inf Inf 0 Inf; Inf Inf 5 0]
        )

        n, m = size(matrix)
        res = floyd_warshall(matrix)
        ref = [0.0 5.0 8.0 Inf; Inf 0.0 3.0 Inf; Inf Inf 0.0 Inf; Inf Inf 5.0 0.0]
        @test res == ref
    end
end
