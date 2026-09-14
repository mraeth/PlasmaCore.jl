# Builds a loader closure that calls `gen(step)` and counts invocations via
# the returned Ref{Int}.
function counting_loader(gen)
    calls = Ref(0)
    loader = step -> begin
        calls[] += 1
        return gen(step)
    end
    return loader, calls
end

@testset "TimeSeries" begin
    @testset "AbstractArray contract + eager first-load" begin
        loader, calls = counting_loader(step -> ScalarField(fill(Float64(step), 2, 2)))
        ts = TimeSeries(loader, collect(1:10))

        @test ts isa AbstractArray{<:ScalarField,1}
        @test size(ts) == (10,)
        @test length(ts) == 10
        @test calls[] == 1  # constructor eagerly loads only the first timestep
    end

    @testset "Lazy caching: loader called once per distinct step" begin
        loader, calls = counting_loader(step -> ScalarField(fill(Float64(step), 2, 2)))
        ts = TimeSeries(loader, collect(1:10))
        @test calls[] == 1

        ts[3]; ts[3]; ts[3]
        @test calls[] == 2  # one real load for step 3, repeats hit the cache

        ts[5]
        @test calls[] == 3

        ts[3]  # already cached
        @test calls[] == 3
    end

    @testset "Sub-series copies already-cached entries, does not live-share" begin
        loader, calls = counting_loader(step -> ScalarField(fill(Float64(step), 2, 2)))
        ts = TimeSeries(loader, collect(1:10))  # calls[] == 1 (step 1 cached)

        ts[3]  # calls[] == 2, step 3 now cached in ts
        @test calls[] == 2

        sub = ts[3:8]  # timesteps [3,4,5,6,7,8]; slicing itself must not load
        @test calls[] == 2

        @test sub[1] === ts[3]  # step 3 was already cached at slice time
        @test calls[] == 2       # no reload through the sub-series

        sub[5]  # step 7, not cached at slice time -> loads through sub's own loader
        @test calls[] == 3
        @test !haskey(ts._cache, 7)  # parent's cache is untouched (not live-shared)
    end

    @testset "Fused spatial+time slicing sugar — 1D-spatial field" begin
        gen(step) = ScalarField(Float64.(step .* (1:5)))
        ts1d = TimeSeries(gen, collect(1:6))

        @test ts1d[2, 3] == ts1d[2][3]
        @test ts1d[2, 3] == 6.0

        sliced = ts1d[2:4, 3]
        @test sliced isa Array{Float64,1}
        @test sliced == [x[3] for x in ts1d[2:4]]
        @test sliced == [6.0, 9.0, 12.0]
    end

    @testset "Fused spatial+time slicing sugar — 2D-spatial field" begin
        gen(step) = ScalarField(Float64(step) .* reshape(collect(1.0:4.0), 2, 2))
        ts2d = TimeSeries(gen, collect(1:5))

        @test ts2d[1, :, :] == ts2d[1]
        @test ts2d[end, :, :] == ts2d[end]

        col1 = ts2d[:, :, 1]
        @test col1 isa TimeSeries
        expected = map(x -> x[:, 1], ts2d)
        @test length(col1) == length(expected)
        @test all(col1[i].data == expected[i].data for i in 1:length(col1))
    end

    @testset "Composed vs fused slicing parity (corrected pairs)" begin
        gen1d(step) = ScalarField(Float64.(step .* (1:5)))
        ts1d = TimeSeries(gen1d, collect(1:6))
        @test ts1d[1:4][:, 1] == ts1d[1:4, 1]

        gen2d(step) = ScalarField(Float64(step) .* reshape(collect(1.0:4.0), 2, 2))
        ts2d = TimeSeries(gen2d, collect(1:6))
        lhs = ts2d[1:4][:, :, 1]
        rhs = ts2d[1:4, :, 1]
        @test length(lhs) == length(rhs)
        @test all(lhs[i].data == rhs[i].data for i in 1:length(lhs))
    end

    @testset "map reconstructs TimeSeries for field-valued f, falls back to Array otherwise" begin
        gen(step) = ScalarField(Float64(step) .* reshape(collect(1.0:4.0), 2, 2))
        ts2d = TimeSeries(gen, collect(1:5))

        centered = map(x -> x .- mean(x), ts2d)
        @test centered isa TimeSeries{<:ScalarField}
        @test all(centered[i].data ≈ ts2d[i].data .- mean(ts2d[i].data) for i in 1:length(ts2d))

        summed = map(x -> sum(x), ts2d)
        @test summed isa Array{Float64,1}
        @test summed == [sum(ts2d[i].data) for i in 1:length(ts2d)]

        gen1d(step) = ScalarField(Float64.(step .* (1:5)))
        ts1d = TimeSeries(gen1d, collect(1:6))
        firsts = map(x -> x[1], ts1d)
        @test firsts isa Array{Float64,1}
        @test firsts == collect(1:6) .* 1.0
    end

    @testset "setindex!" begin
        loader, calls = counting_loader(step -> ScalarField(fill(Float64(step), 2, 2)))
        ts = TimeSeries(loader, collect(1:5))
        @test calls[] == 1

        replacement = ScalarField(fill(999.0, 2, 2))
        ts[2] = replacement
        @test ts[2] === replacement
        @test calls[] == 1  # setindex! must not touch the loader
    end
end
