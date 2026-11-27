# Captured when this module is compiled, so a later checkout change cannot alter
# the reported identity of already loaded code. Registered source trees may have
# no Git metadata; do not accidentally report an enclosing, unrelated repository.
function _software_version()
    root = dirname(@__DIR__)
    version = Base.pkgversion(@__MODULE__)
    git_commit = nothing
    git_dirty = nothing
    try
        tracked = readchomp(pipeline(
            `git -C $root ls-files --error-unmatch -- src/FilterGraphs.jl`;
            stderr=devnull))
        if tracked == "src/FilterGraphs.jl"
            git_commit = readchomp(pipeline(`git -C $root rev-parse HEAD`; stderr=devnull))
            git_dirty = !isempty(readchomp(pipeline(
                `git -C $root status --porcelain --untracked-files=normal`; stderr=devnull)))
        end
    catch
        # Git is optional, including when loading a registered package or archive.
    end
    return (;version, git_commit, git_dirty)
end

const _SOFTWARE_VERSION = _software_version()

"""
    software_version()

Return `(version, git_commit, git_dirty)` for this FilterGraphs module.
The immutable snapshot is captured at compilation/precompilation, never during
processing. `version` is the package version; Git fields are `nothing` when
source Git metadata is unavailable. An existing compiled cache retains its build
identity even if Git HEAD subsequently changes. A dirty build identifies a
modified checkout, not an exact reproducible source tree. This query describes
FilterGraphs, not every loaded dependency or the linked PipeWire library.
"""
software_version() = _SOFTWARE_VERSION
