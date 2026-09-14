@testset "ScalarField" begin
    @testset "AbstractArray contract" begin
        data = reshape(collect(1.0:12.0), 3, 4)
        sf = ScalarField(data)

        @test sf isa AbstractArray{Float64,2}
        @test size(sf) == (3, 4)
        @test sf[2, 3] == data[2, 3]

        sf[2, 3] = 99.0
        @test sf.data[2, 3] == 99.0
        sf[2, 3] = data[2, 3]  # restore

        @test IndexStyle(typeof(sf)) == IndexStyle(Matrix{Float64})
    end

    @testset "Slicing returns ScalarField, not plain Array" begin
        data = reshape(collect(1.0:12.0), 3, 4)
        sf = ScalarField(data)

        row_slice = sf[1:2, :]
        @test row_slice isa ScalarField
        @test row_slice.data == data[1:2, :]

        col_slice = sf[:, 2]
        @test col_slice isa ScalarField
        @test ndims(col_slice) == 1
        @test col_slice.data == data[:, 2]
    end

    @testset "Reductions, broadcasting, map, collect, Array" begin
        data = reshape(collect(1.0:12.0), 3, 4)
        sf = ScalarField(data)

        @test mean(sf) == mean(data)
        @test sum(sf) == sum(data)

        centered = sf .- mean(sf)
        @test centered isa ScalarField
        @test centered.data == data .- mean(data)

        doubled = map(x -> 2x, sf)
        @test doubled isa ScalarField
        @test doubled.data == 2 .* data

        @test collect(sf) isa Array{Float64,2}
        @test collect(sf) == data
        @test Array(sf) == data
    end

    @testset "Regression: ScalarField * ScalarField is elementwise, not matmul" begin
        # Chosen so that elementwise (Hadamard) product and real matrix
        # multiplication give different results — otherwise this test could
        # pass vacuously even if `*` silently fell back to LinearAlgebra's
        # generic AbstractMatrix*AbstractMatrix (real matmul).
        a = ScalarField(Float64.(reshape(1:9, 3, 3)))
        b = ScalarField(Float64.(reshape(9:-1:1, 3, 3)))
        @assert a.data .* b.data != a.data * b.data "test precondition failed: Hadamard and matmul coincide for this data"

        prod = a * b
        @test prod isa ScalarField
        @test prod.data == a.data .* b.data
        @test prod.data != a.data * b.data
    end

    @testset "Division by a Number" begin
        sf = ScalarField(reshape(collect(1.0:12.0), 3, 4))
        halved = sf / 2.0
        @test halved isa ScalarField
        @test halved.data == sf.data ./ 2.0
    end
end
