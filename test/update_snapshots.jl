# Maintain selected copied inputs through the canonical source and receipt writer.
using SHA, TOML
const ROOT=normpath(joinpath(@__DIR__,".."))
isempty(ARGS) && error("supply canonical repository-relative source paths")
packages=Set(replace(arg,"--package="=>"") for arg in ARGS if startswith(arg,"--package="))
selected=Set(arg for arg in ARGS if !startswith(arg,"--package="))
updated=Set{String}()
for (directory,_,files) in walkdir(joinpath(ROOT,"julia"))
    relative=relpath(directory,joinpath(ROOT,"julia"))
    isempty(packages) || first(splitpath(relative)) in packages || continue
    for filename in files
        endswith(filename,"snapshot-manifest.toml") || continue
        path=joinpath(directory,filename)
        raw=read(path,String)
        blocks=split(raw,"[[snapshot]]")
        changed=false
        for index in 2:length(blocks)
            entry=only(TOML.parse("[[snapshot]]"*blocks[index])["snapshot"])
            original_source=entry["source"]
            entry["source"]=replace(original_source,
                "julia/JuliaFilterGraph/"=>"julia/FilterGraphs/",
                "julia/FilterGraphIsland/test/"=>"julia/FilterGraphPipeWire/test/application/")
            entry["source"] in selected || continue
            source=read(joinpath(ROOT,entry["source"]))
            commentable=splitext(entry["source"])[2] in (".jl",".conf",".toml",".yaml",".yml")
            header=commentable ? "# Generated snapshot from $(entry["source"]).\n# Regenerate with julia test/update_snapshots.jl $(entry["source"]).\n" : ""
            emitted=vcat(Vector{UInt8}(codeunits(header)),source)
            write(joinpath(directory,entry["path"]),emitted)
            digest=bytes2hex(sha256(emitted))
            block=replace(blocks[index], original_source=>entry["source"])
            block=replace(block,r"sha256 = \"[0-9a-f]+\""=>"sha256 = \"$digest\"")
            if haskey(entry,"generated_header_lines")
                block=replace(block,r"generated_header_lines = [0-9]+"=>"generated_header_lines = $(commentable ? 2 : 0)")
            elseif commentable
                block=replace(block,"sha256 = \"$digest\""=>"sha256 = \"$digest\"\ngenerated_header_lines = 2")
            end
            blocks[index]=block
            push!(updated,entry["source"])
            changed=true
        end
        if changed
            output=join(blocks,"[[snapshot]]")
            startswith(output,"# Generated") || (output="# Generated snapshot receipts; canonical sources are listed below.\n# Regenerate selected copies and hashes with julia test/update_snapshots.jl <source>...\n"*output)
            write(path,output)
        end
    end
end
selected==updated || error("some selected sources have no snapshot receipt: $(setdiff(selected,updated))")
