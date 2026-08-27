
abstract type Distribution end

struct DistributionGridImpl{NX,NV,ID,SF<:ScalarField} <: Distribution
    _field::SF
end

const DistributionGrid{DT,NX,NV,NXNV,ID,AT} = DistributionGridImpl{NX,NV,ID,ScalarField{DT,NXNV,AT}}

const DistributionGrid1d1v{T,ID,AT} = DistributionGrid{T,1,1,2,ID,AT}
const DistributionGrid1d2v{T,ID,AT} = DistributionGrid{T,1,2,3,ID,AT}
const DistributionGrid2d2v{T,ID,AT} = DistributionGrid{T,2,2,4,ID,AT}

function Base.getproperty(f::DistributionGridImpl, sym::Symbol)
    sym === :data  && return getfield(f, :_field).data
    sym === :field && return getfield(f, :_field)
    return getfield(f, sym)
end
