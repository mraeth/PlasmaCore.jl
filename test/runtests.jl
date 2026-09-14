using Test
using PlasmaCore
using PlasmaCore: ScalarField, VectorField, MatrixField, TensorField, TimeSeries,
    Grid, SimulationTime, Polar,
    grad, div, curl, differentiate, ncomponents, outer_product,
    empty_scalarfield, empty_vectorfield, empty_matrixfield, zero_vectorfield_like,
    zero_scalarfield_like, zero_vectorfield3, vectorfield_from_spatial_components,
    background_field, Moments, FieldSolution,
    backend_copy, use_cpu!, backend_synchronize!, DistributionGridImpl,
    advance!, continue_advection
using Statistics: mean
using LinearAlgebra

@testset "PlasmaCore.jl" begin
    include("test_scalarfield.jl")
    include("test_tensorfield.jl")
    include("test_timeseries.jl")
    include("test_spectral_operators.jl")
    include("test_grid_time_indexing.jl")
    include("test_execution_and_solver.jl")
end
