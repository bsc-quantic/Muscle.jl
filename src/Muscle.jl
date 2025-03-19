module Muscle

import EinExprs: inds
using Compat: @compat

include("Utils/Utils.jl")

include("Index.jl")
export NamedIndex
export Site, issite, site, @site_str
export Bond, isbond, bond, @bond_str
export Plug, isplug, plug, isdual, @plug_str

@compat public Moment

include("Tensor.jl")
export Tensor

# rexports from EinExprs
export inds

end
