using Pkg
using TOML

Pkg.setprotocol!(domain="github.com", protocol="https")
Pkg.Registry.add("General")
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/DarrylGamroth/PackageRegistry.git"))

project_file = Base.active_project()
project = TOML.parsefile(project_file)
dependencies = project["deps"]
sources = get!(Dict{String,Any}, project, "sources")
repository = normpath(joinpath(@__DIR__, "..", ".."))
for package in ("FilterGraphAlgorithms", "FilterGraphs", "FilterGraphPipeWire")
    if haskey(dependencies, package)
        sources[package] = Dict("path" => joinpath(repository, "julia", package))
    end
end

dependencies["AdaptiveOpticsCalibration"] = "3c8b5851-926e-4ebb-af30-f8544b98d45f"
sources["AdaptiveOpticsCalibration"] = Dict(
    "url" => "https://github.com/DarrylGamroth/AdaptiveOpticsCalibration.git",
    "rev" => "v0.17.7",
)
open(io -> TOML.print(io, project), project_file, "w")
Pkg.instantiate()

package = Base.PkgId(Base.UUID(dependencies["AdaptiveOpticsCalibration"]), "AdaptiveOpticsCalibration")
installed = Pkg.dependencies()[package.uuid]
@assert installed.version == v"0.17.7" "CPU CI must use AdaptiveOpticsCalibration v0.17.7"
@assert installed.tree_hash == "b44915b9b588b830573dde2eb8fdae9c90a0734c" "AdaptiveOpticsCalibration v0.17.7 tree changed"
