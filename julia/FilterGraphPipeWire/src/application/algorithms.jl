function capture_device(backend::AbstractString)
    Base.require(Main, :KernelAbstractions)
    if backend == "amdgpu"
        package = Base.require(Main, :AMDGPU)
        Base.invokelatest(package.functional) ||
            error("AMDGPU is not functional on this host")
        Base.invokelatest(package.allowscalar, false)
        return Base.invokelatest(package.device)
    end
    package = Base.require(Main, :CUDA)
    Base.invokelatest(package.functional) || error("CUDA is not functional on this host")
    Base.invokelatest(package.allowscalar, false)
    return Base.invokelatest(package.device)
end

function load_algorithm_package(name::AbstractString)
    Base.isidentifier(name) || usage_error(
        "Algorithm package must be one Julia module identifier: $name",
    )
    provider = Base.require(Main, Symbol(name))
    isdefined(provider, :algorithms) || throw(
        ArgumentError("Algorithm package $name does not define algorithms()"),
    )
    declarations = Base.invokelatest(getproperty(provider, :algorithms))
    declarations isa Tuple || throw(
        ArgumentError("$name.algorithms() must return a Tuple"),
    )
    return declarations
end

"""
    algorithm_declarations(; packages=(), algorithms=(), builtins=true)

Load the selected built-in and application Algorithm declarations, removing
duplicates while preserving selection order. Optional providers must be available
in the active application environment and expose `algorithms()` or the selected
exported declaration.
"""
function algorithm_declarations(; packages=(), algorithms=(), builtins=true)
    declarations = builtins ? FilterGraphAlgorithms.algorithms() : ()
    for package in packages
        declarations = (declarations..., load_algorithm_package(package)...)
    end
    for name in algorithms
        declarations = (declarations..., load_algorithm_declaration(name))
    end
    return Tuple(unique(declarations))
end

function algorithm_identifier(name::AbstractString)
    parts = split(name, '.'; keepempty=true)
    length(parts) == 2 && all(Base.isidentifier, parts) || usage_error(
        "--algorithm must name MODULE.EXPORTED_DECLARATION: $name",
    )
    return Symbol(parts[1]), Symbol(parts[2])
end

function load_algorithm_declaration(name::AbstractString)
    package, identifier = algorithm_identifier(name)
    provider = Base.require(Main, package)
    isdefined(provider, identifier) && Base.isexported(provider, identifier) ||
        throw(ArgumentError("Algorithm $name must be an exported declaration"))
    return validate_algorithm_declaration(getproperty(provider, identifier), name)
end

function validate_algorithm_declaration(declaration::Type{<:FilterGraphAlgorithms.AbstractAlgorithmDeclaration}, name)
    isconcretetype(declaration) || throw(ArgumentError(
        "$name must be a concrete Algorithm declaration type",
    ))
    return declaration
end
validate_algorithm_declaration(declaration, name) = throw(ArgumentError(
    "$name must be an Algorithm declaration type",
))

_algorithm_declarations(options) = algorithm_declarations(;
    packages=options.packages, algorithms=options.algorithms, builtins=options.builtins)
