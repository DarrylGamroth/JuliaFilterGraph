@testset "software identity" begin
    version = FilterGraphs.software_version()
    @test version.version == Base.pkgversion(FilterGraphs)
    @test Base.PkgId(FilterGraphs).uuid == Base.UUID("8053475d-7091-4a8a-adbc-8ed688b6da8d")
    @test isnothing(version.git_commit) || occursin(r"^[0-9a-f]{40,64}$", version.git_commit)
    @test version.git_dirty isa Union{Nothing,Bool}
    @test FilterGraphs.software_version() === version
    # Diagnostics are not stored on Graphs or attached to results.
    @test :revision ∉ fieldnames(PreparedGraph)
    @test !isdefined(FilterGraphs, :ConfigurationReceipt)
    @test !isdefined(FilterGraphs, :PreparedSampleConfiguration)
end

@testset "software identity survives checkout changes and absent Git" begin
    source = joinpath(dirname(pathof(FilterGraphs)), "version.jl")
    snapshot(root) = begin
        module_ = Module(gensym(:SoftwareVersionTest))
        Base.include(module_, joinpath(root, "src", "version.jl"))
        module_
    end
    query(module_) = Base.invokelatest(() -> module_.software_version())
    mktempdir() do root
        mkpath(joinpath(root, "src"))
        cp(source, joinpath(root, "src", "version.jl"))
        write(joinpath(root, "src", "FilterGraphs.jl"), "# fixture\n")
        withenv("PATH" => "") do
            @test isnothing(query(snapshot(root)).git_commit)
            @test isnothing(query(snapshot(root)).git_dirty)
        end
        if !isnothing(Sys.which("git"))
            run(pipeline(`git init -q $root`; stderr=devnull))
            run(`git -C $root add src`)
            run(pipeline(`git -C $root -c user.name=Test -c user.email=test@example.invalid commit -qm initial`; stderr=devnull))
            loaded = snapshot(root)
            built = query(loaded)
            @test built.git_commit == readchomp(`git -C $root rev-parse HEAD`)
            @test built.git_dirty === false
            write(joinpath(root, "src", "FilterGraphs.jl"), "# changed\n")
            @test query(snapshot(root)).git_dirty === true
            run(`git -C $root add src`)
            run(pipeline(`git -C $root -c user.name=Test -c user.email=test@example.invalid commit -qm changed`; stderr=devnull))
            @test query(loaded) === built
            @test query(snapshot(root)).git_commit != built.git_commit
            # A source archive inside an unrelated repository must stay unknown.
            archive = joinpath(root, "archive")
            mkpath(joinpath(archive, "src"))
            cp(source, joinpath(archive, "src", "version.jl"))
            write(joinpath(archive, "src", "FilterGraphs.jl"), "# untracked\n")
            @test isnothing(query(snapshot(archive)).git_commit)
        end
    end
end
