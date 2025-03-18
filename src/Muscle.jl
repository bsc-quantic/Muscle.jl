module Muscle

using Compat: @compat

include("Index.jl")
export NamedIndex
export Site, issite, site, @site_str
export Bond, isbond, bond
export Plug, isplug, plug, isdual, @plug_str

@compat public Moment

end
