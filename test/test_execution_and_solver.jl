@testset "Execution & field_solver (CPU only)" begin
    use_cpu!()

    grid = Grid([0.0, 0.0], [2pi, 2pi], [8, 8], 2)

    @testset "Allocation helpers" begin
        sf = empty_scalarfield(grid)
        @test sf isa ScalarField
        @test size(sf) == (8, 8)
        @test all(iszero, sf.data)

        vf = empty_vectorfield(grid, 2)
        @test vf isa VectorField
        @test length(vf) == 2
        @test all(all(iszero, c.data) for c in vf)

        mf = empty_matrixfield(grid, 2)
        @test mf isa MatrixField
        @test size(mf) == (2, 2)
        @test all(all(iszero, mf[i, j].data) for i in 1:2, j in 1:2)
    end

    @testset "zero_vectorfield_like independence" begin
        vf = empty_vectorfield(grid, 2)
        vf[1].data .= 1.0
        z = zero_vectorfield_like(vf)
        @test all(iszero, z[1].data)
        z[1].data[1, 1] = 5.0
        @test vf[1].data[1, 1] == 1.0  # independent storage
    end

    @testset "Moments dimension check" begin
        rho8 = empty_scalarfield(grid)
        J8 = empty_vectorfield(grid, 2)
        @test Moments(rho8, J8) isa Moments

        grid4 = Grid([0.0, 0.0], [2pi, 2pi], [4, 4], 2)
        Jmismatch = empty_vectorfield(grid4, 2)
        @test_throws DimensionMismatch Moments(rho8, Jmismatch)
    end

    @testset "FieldSolution" begin
        E = empty_vectorfield(grid, 2)
        B = empty_vectorfield(grid, 2)
        fs = FieldSolution(E, B)
        @test fs.phi isa ScalarField
        @test all(iszero, fs.phi.data)
        @test size(fs.phi) == size(E[1])
    end

    @testset "vectorfield_from_spatial_components" begin
        c1 = empty_scalarfield(grid)
        c2 = empty_scalarfield(grid)
        @test length(vectorfield_from_spatial_components([c1])) == 3
        @test length(vectorfield_from_spatial_components([c1, c2])) == 3

        @test_throws ArgumentError vectorfield_from_spatial_components(ScalarField[])
        c3 = empty_scalarfield(grid)
        c4 = empty_scalarfield(grid)
        @test_throws ArgumentError vectorfield_from_spatial_components([c1, c2, c3, c4])
    end

    @testset "zero_vectorfield3 / background_field" begin
        z3 = zero_vectorfield3(grid)
        @test length(z3) == 3
        @test all(all(iszero, c.data) for c in z3)

        bg = background_field(grid)  # default Bdir=3
        @test all(iszero, bg[1].data)
        @test all(iszero, bg[2].data)
        @test all(==(grid.b0), bg[3].data)

        badgrid = Grid([0.0, 0.0], [2pi, 2pi], [8, 8], 2, 1.0, 0)
        @test_throws ArgumentError background_field(badgrid)
    end

    @testset "backend_copy / backend_synchronize!" begin
        sf = empty_scalarfield(grid)
        sf.data[1, 1] = 42.0
        sfcopy = backend_copy(sf)
        @test sfcopy.data == sf.data
        sfcopy.data[1, 1] = 0.0
        @test sf.data[1, 1] == 42.0  # independent storage

        vf = empty_vectorfield(grid, 2)
        vfcopy = backend_copy(vf)
        @test vfcopy isa VectorField
        @test all(vfcopy[i].data == vf[i].data for i in 1:2)

        mf = empty_matrixfield(grid, 2)
        mfcopy = backend_copy(mf)
        @test mfcopy isa MatrixField

        @test isnothing(backend_synchronize!())
    end

    @testset "DistributionGridImpl smoke" begin
        sf = ScalarField(zeros(4, 4, 4, 4))
        dg = DistributionGridImpl{2,2,:test,typeof(sf)}(sf)
        @test dg.data === sf.data
        @test dg.field === sf
    end
end
