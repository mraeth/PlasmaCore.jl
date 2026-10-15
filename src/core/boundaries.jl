# Boundary conditions along one axis, shared by the derivative operators here and the
# advection in consumer packages (bslLD).

"""
    Periodic()

Periodic axis: values beyond the last node wrap around to the first.
"""
struct Periodic end

"""
    Mirror()

Even reflection about the first and last node (the node fold `… 3, 2 | 1 … n | n-1, n-2 …`),
as for the density of a specularly reflected distribution. The walls sit on the end nodes.
"""
struct Mirror end
