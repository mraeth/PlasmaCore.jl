abstract type AbstractFieldSolver end

struct Moments{
    SF<:ScalarField,
    VF<:Union{Nothing,VectorField},
    PF<:Union{Nothing,VectorField},
}
    rho::SF
    J::VF
    Pi_diff::PF

    function Moments(
        rho::SF,
        J::VF = nothing,
        Pi_diff::PF = nothing,
    ) where {SF<:ScalarField,VF<:Union{Nothing,VectorField},PF<:Union{Nothing,VectorField}}
        if J !== nothing
            axes(rho.data) == axes(J[1].data) ||
                throw(DimensionMismatch("Moments rho and J must share spatial axes"))
        end
        if Pi_diff !== nothing
            axes(rho.data) == axes(Pi_diff[1].data) ||
                throw(DimensionMismatch("Moments rho and Pi_diff must share spatial axes"))
        end
        return new{SF,VF,PF}(rho, J, Pi_diff)
    end
end

struct FieldSolution{VF<:VectorField,SF<:ScalarField}
    E::VF
    B::VF
    Enew::VF   # E^{n+1}, written by solve_fields!
    phi::SF    # electrostatic potential; zero for solvers where phi is not meaningful
end

function FieldSolution(E::VF, B::VF) where {VF<:VectorField}
    phi = zero_scalarfield_like(E[1])
    return FieldSolution{VF,typeof(phi)}(E, B, zero_vectorfield_like(E), phi)
end

function zero_scalarfield_like(field::ScalarField)
    return ScalarField(zero.(field.data))
end

function vectorfield_from_spatial_components(components::AbstractVector{<:ScalarField})
    ncomp = length(components)
    1 <= ncomp <= 3 ||
        throw(ArgumentError("field solvers support between 1 and 3 physical components"))

    padded = ScalarField[components...]
    if ncomp < 3
        zero_component = zero_scalarfield_like(first(components))
        append!(padded, [zero_component for _ = 1:(3-ncomp)])
    end

    return VectorField(padded)
end

function zero_vectorfield3(grid::Grid)
    dims = Tuple(length.(grid.xaxes))
    return VectorField([zeros(Float64, dims) for _ = 1:3])
end

function background_field(grid::Grid)
    1 <= grid.Bdir <= 3 || throw(ArgumentError("grid.Bdir must be between 1 and 3"))

    dims = Tuple(length.(grid.xaxes))
    components = [zeros(Float64, dims) for _ = 1:3]
    components[grid.Bdir] .= grid.b0
    return VectorField(components)
end
