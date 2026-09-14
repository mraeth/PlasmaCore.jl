@testset "Grid" begin
    @testset "Cartesian — no velocity axes" begin
        g = Grid([0.0, 0.0], [2pi, 2pi], [8, 8], 2)
        @test length(g.xaxes) == 2
        @test length.(g.xaxes) == (8, 8)
        @test all(isapprox.(g.delta, [2pi / 8, 2pi / 8]))
        @test isempty(g.vaxes)
    end

    @testset "Cartesian — with a velocity axis" begin
        g = Grid([0.0, 0.0, 0.0], [1.0, 1.0, 10.0], [4, 4, 5], 2)
        @test length(g.xaxes) == 2
        @test length(g.vaxes) == 1
        @test length(g.vaxes[1]) == 5 + 1
        @test step(g.vaxes[1]) ≈ 10.0 / 5
    end

    @testset "Polar" begin
        g = Grid([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 5.0, 2pi], [4, 4, 10, 8], 2; type = Polar)
        @test g isa PlasmaCore.PolarGrid
        @test length(g.vaxes) == 2
        @test step(g.vaxes[1]) ≈ 5.0 / 10
        @test step(g.vaxes[2]) ≈ 2pi / 8
    end

    @testset "outer_product" begin
        v1 = [1.0, 2.0]
        v2 = [10.0, 20.0, 30.0]
        result = outer_product([v1, v2])
        @test size(result) == (2, 3)
        for i in 1:2, j in 1:3
            @test result[i, j] == v1[i] * v2[j]
        end
    end
end

@testset "SimulationTime" begin
    t = SimulationTime(0.1, 1.0)
    @test t[1] == 0.0
    @test t[2] ≈ 0.1
    @test t[3] ≈ 0.2

    step0 = t.step
    T0 = t.current_T
    advance!(t)
    @test t.step == step0 + 1
    @test t.current_T ≈ T0 + 0.1

    @test continue_advection(t) == (t.current_T < t.final_T && t.step < t.nmax)

    v1, state1 = iterate(t)
    @test v1 == 0.0
    v2, _ = iterate(t, state1)
    @test v2 ≈ 0.1
end

@testset "index_*_to_* helpers round trip" begin
    sizes = (3, 4)
    for i in 1:prod(sizes)
        nd = PlasmaCore.index_1d_to_nd(i, sizes)
        @test PlasmaCore.index_nd_to_1d(nd, sizes) == i
    end

    sizesX = (2, 3)
    sizesY = (2, 2)
    for ixflat in 1:prod(sizesX), iyflat in 1:prod(sizesY)
        indicesX = PlasmaCore.index_1d_to_nd(ixflat, sizesX)
        indicesY = PlasmaCore.index_1d_to_nd(iyflat, sizesY)
        combined = PlasmaCore.index_combined_to_1d(indicesX, indicesY, sizesX, sizesY)
        back = PlasmaCore.index_1d_to_combined(combined, sizesX, sizesY)
        @test back == (indicesX, indicesY)
    end
end
