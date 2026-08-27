struct Grid{T,XT,VT,MT,ID}
    xaxes::XT
    vaxes::VT
    max::MT
    min::MT
    delta::MT
    b0::T
    Bdir::Int
end

abstract type Cart end
abstract type Polar end

const CartGrid = Grid{T,XT,VT,MT,Cart} where {T,XT,VT,MT}
const PolarGrid = Grid{T,XT,VT,MT,Polar} where {T,XT,VT,MT}

function Adapt.adapt_structure(to, grid::Grid{T,XT,VT,MT,ID}) where {T,XT,VT,MT,ID}
    xaxes = Adapt.adapt(to, grid.xaxes)
    vaxes = Adapt.adapt(to, grid.vaxes)
    maxv = Adapt.adapt(to, grid.max)
    minv = Adapt.adapt(to, grid.min)
    delta = Adapt.adapt(to, grid.delta)
    b0 = Adapt.adapt(to, grid.b0)

    return Grid{typeof(b0),typeof(xaxes),typeof(vaxes),typeof(maxv),ID}(
        xaxes,
        vaxes,
        maxv,
        minv,
        delta,
        b0,
        grid.Bdir,
    )
end

_axis_tuple(axes::Vector) = Tuple(axes)
_meta_vector(values::AbstractVector{T}) where {T} = SVector{length(values),T}(values)

function Grid(
    etaMin::Vector{Float64},
    etaMax::Vector{Float64},
    N::Vector{Int64},
    nx::Int64,
    b0 = 1.0::Float64,
    Bdir = 3::Int;
    type = Cart,
)
    if type == Cart
        delta = (etaMax .- etaMin) ./ N
        xaxes = [range(etaMin[i], step = delta[i], length = N[i]) for i = 1:nx]
        vaxes = [
            range(etaMin[i], step = delta[i], length = N[i] + 1) for
            i = (nx+1):length(etaMin)
        ]

        return Grid{
            typeof(b0),
            typeof(_axis_tuple(xaxes)),
            typeof(_axis_tuple(vaxes)),
            typeof(_meta_vector(etaMax)),
            Cart,
        }(
            _axis_tuple(xaxes),
            _axis_tuple(vaxes),
            _meta_vector(etaMax),
            _meta_vector(etaMin),
            _meta_vector(delta),
            b0,
            Bdir,
        )
    else
        vpmax = etaMax[nx+1]
        deltavp = vpmax / N[nx+1]
        deltaphi = 2pi / N[nx+2]

        delta = vcat((etaMax[1:nx] .- etaMin[1:nx]) ./ N[1:nx], [deltavp, deltaphi])

        xaxes = [range(etaMin[i], step = delta[i], length = N[i]) for i = 1:nx]

        vaxes = (
            range(deltavp, step = deltavp, length = N[nx+1]),
            range(0.0, step = deltaphi, length = N[nx+2]),
        )

        return Grid{
            typeof(b0),
            typeof(_axis_tuple(xaxes)),
            typeof(vaxes),
            typeof(_meta_vector(etaMax)),
            Polar,
        }(
            _axis_tuple(xaxes),
            vaxes,
            _meta_vector(etaMax),
            _meta_vector(etaMin),
            _meta_vector(delta),
            b0,
            Bdir,
        )
    end
end

outer_product(vs) =
    .*([reshape(vs[d], (ntuple(Returns(1), d - 1)..., :)) for d = 1:length(vs)]...)
