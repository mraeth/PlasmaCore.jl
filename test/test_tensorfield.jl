@testset "TensorField / VectorField / MatrixField" begin
    @testset "Component indexing and row-major flat layout" begin
        vf = VectorField([ScalarField(fill(1.0, 2, 2)),
                           ScalarField(fill(2.0, 2, 2)),
                           ScalarField(fill(3.0, 2, 2))])
        @test size(vf) == (3,)
        @test length(vf) == 3
        @test vf[1].data == fill(1.0, 2, 2)
        @test vf[2].data == fill(2.0, 2, 2)
        @test vf[3].data == fill(3.0, 2, 2)

        raw = Matrix{ScalarField}(undef, 2, 2)
        raw[1, 1] = ScalarField(fill(1.0, 2, 2))
        raw[1, 2] = ScalarField(fill(2.0, 2, 2))
        raw[2, 1] = ScalarField(fill(3.0, 2, 2))
        raw[2, 2] = ScalarField(fill(4.0, 2, 2))
        mf = MatrixField(raw)
        NF = 4
        NS = isqrt(NF)
        @test size(mf) == (2, 2)
        for i in 1:NS, j in 1:NS
            @test mf[i, j] === mf.data[(i-1)*NS+j]
        end
        @test mf[1, 1].data == fill(1.0, 2, 2)
        @test mf[1, 2].data == fill(2.0, 2, 2)
        @test mf[2, 1].data == fill(3.0, 2, 2)
        @test mf[2, 2].data == fill(4.0, 2, 2)
        @test length(mf) == 4
    end

    @testset "Iteration order matches flat NTuple order" begin
        vf = VectorField([ScalarField(fill(Float64(i), 2, 2)) for i in 1:3])
        @test collect(vf) == [vf.data...]

        raw = Matrix{ScalarField}(undef, 2, 2)
        raw[1, 1] = ScalarField(fill(1.0, 2, 2)); raw[1, 2] = ScalarField(fill(2.0, 2, 2))
        raw[2, 1] = ScalarField(fill(3.0, 2, 2)); raw[2, 2] = ScalarField(fill(4.0, 2, 2))
        mf = MatrixField(raw)
        # `mf` is a genuine 2D AbstractArray, so a comprehension over it
        # preserves its (2,2) shape; `vec` reads that shape back out in the
        # same linear order the elements were iterated/stored in, which is
        # what should match the flat NTuple storage order.
        @test vec(collect(mf)) == [mf.data...]
    end

    @testset "Arithmetic: +, -, scalar * (both operand orders)" begin
        # Deliberately not testing `.+`/`.-` dot-broadcast: TensorField has no
        # BroadcastStyle, so dot-broadcast falls back to a plain Array of
        # ScalarField rather than reconstructing a TensorField (by design,
        # deferred per the restructuring notes).
        v1 = VectorField([ScalarField([3.0]), ScalarField([4.0])])
        v2 = VectorField([ScalarField([10.0]), ScalarField([20.0])])

        vsum = v1 + v2
        @test vsum isa VectorField
        @test vsum[1].data == [13.0] && vsum[2].data == [24.0]

        vdiff = v1 - v2
        @test vdiff isa VectorField
        @test vdiff[1].data == [-7.0] && vdiff[2].data == [-16.0]

        @test (2.0 * v1) isa VectorField
        @test (2.0 * v1)[1].data == [6.0] && (2.0 * v1)[2].data == [8.0]
        @test (v1 * 2.0)[1].data == [6.0] && (v1 * 2.0)[2].data == [8.0]

        raw1 = reshape([ScalarField([Float64(k)]) for k in 1:4], 2, 2)
        raw2 = reshape([ScalarField([10.0 * k]) for k in 1:4], 2, 2)
        mf1 = MatrixField(raw1)
        mf2 = MatrixField(raw2)

        msum = mf1 + mf2
        @test msum isa MatrixField
        for i in 1:2, j in 1:2
            @test msum[i, j].data == mf1[i, j].data .+ mf2[i, j].data
        end

        @test (3.0 * mf1) isa MatrixField
        for i in 1:2, j in 1:2
            @test (3.0 * mf1)[i, j].data == 3.0 .* mf1[i, j].data
            @test (mf1 * 3.0)[i, j].data == 3.0 .* mf1[i, j].data
        end
    end

    @testset "Rotation / transpose / matmul (hand-computed)" begin
        # 1-element spatial arrays so every product is trivially hand-checkable.
        v = VectorField([ScalarField([3.0]), ScalarField([4.0])])

        raw = Matrix{ScalarField}(undef, 2, 2)
        raw[1, 1] = ScalarField([1.0]); raw[1, 2] = ScalarField([2.0])
        raw[2, 1] = ScalarField([3.0]); raw[2, 2] = ScalarField([4.0])
        mf = MatrixField(raw)

        R = [0.0 -1.0; 1.0 0.0]  # 90-degree rotation

        for Rop in (R, transpose(R), R')
            Rv = Rop * v
            @test Rv isa VectorField
            expected1 = Rop[1, 1] * 3.0 + Rop[1, 2] * 4.0
            expected2 = Rop[2, 1] * 3.0 + Rop[2, 2] * 4.0
            @test Rv[1].data == [expected1]
            @test Rv[2].data == [expected2]
        end

        Rm = R * mf
        @test Rm isa MatrixField
        @test Rm[1, 1].data == [-3.0]
        @test Rm[1, 2].data == [-4.0]
        @test Rm[2, 1].data == [1.0]
        @test Rm[2, 2].data == [2.0]

        mR = mf * R
        @test mR isa MatrixField
        @test mR[1, 1].data == [2.0]
        @test mR[1, 2].data == [-1.0]
        @test mR[2, 1].data == [4.0]
        @test mR[2, 2].data == [-3.0]

        mv = mf * v
        @test mv isa VectorField
        @test mv[1].data == [11.0]  # 1*3 + 2*4
        @test mv[2].data == [25.0]  # 3*3 + 4*4

        mt = transpose(mf)
        @test mt isa MatrixField
        for i in 1:2, j in 1:2
            @test mt[i, j].data == mf[j, i].data
        end
        @test adjoint(mf) isa MatrixField
        for i in 1:2, j in 1:2
            @test adjoint(mf)[i, j].data == transpose(mf)[i, j].data
        end
    end

    @testset "mean(vf::VectorField) reduces components to a ScalarField" begin
        vf = VectorField([ScalarField(fill(2.0, 2, 2)),
                           ScalarField(fill(4.0, 2, 2)),
                           ScalarField(fill(6.0, 2, 2))])
        m = mean(vf)
        @test m isa ScalarField
        @test m.data == fill(4.0, 2, 2)
    end

    @testset "Constructor validation" begin
        @test_throws ArgumentError VectorField(ScalarField[])

        mismatched = [ScalarField(fill(1.0, 2, 2)), ScalarField(fill(1.0, 3, 3))]
        @test_throws DimensionMismatch VectorField(mismatched)

        nonsquare = [ScalarField(fill(Float64(k), 1, 1)) for k in 1:6]
        @test_throws ArgumentError MatrixField(reshape(nonsquare, 2, 3))
    end
end
