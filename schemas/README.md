# Scientific schemas

[`catalog.toml`](catalog.toml) is the authority for public, versioned scientific
ndarray schemas. It records each quantity name, Julia element type, schema
version, and scientific definition once. The generator constructs opaque identifiers as
`pipewireao/<name>-<type>/<version>`; `Float32` and `Float64` declarations have
distinct identifiers. Existing controller-coordinate quantities do not imply a
physical unit.

`FilterGraphAlgorithms` exposes the matching constants and docstrings from the
generated [`schemas.jl`](../julia/FilterGraphAlgorithms/src/schemas.jl).
Optional scalar channel declarations and their source publication are defined in
the [source channel contract](../docs/source-channels.md).

From the repository root, regenerate the Julia source after changing the catalog
and check that it is current with:

```sh
julia scripts/generate_schemas.jl
julia scripts/generate_schemas.jl --check
# Generate consumer projections from this same catalog:
julia scripts/generate_schemas.jl --rtc /path/to/rtc --gui /path/to/gui --hil /path/to/hil
julia scripts/generate_schemas.jl --check --rtc /path/to/rtc --gui /path/to/gui --hil /path/to/hil
```

The repository tests also require each exported schema constant to have exactly
one catalog entry and require the generated source to match the catalog.
The optional outputs are RTC `data/schemas.yaml`, GUI `src/schemas.rs`, and HIL
`src/schemas.jl`. They are generated projections; edit this catalog and regenerate.
RTC records contain the ID, Julia element type, and SPA ndarray element-type name.
Detector-frame schemas also declare their frame encoding where applicable.

`FilterGraphAlgorithms.schema_element_type(id)` returns the declared Julia type
for a known ID and `nothing` for an external schema. Scientific Port declarations
use this lookup, and `AlgorithmFormat` rejects known schema/type mismatches.
External/custom versioned schema identifiers remain supported; their owner defines
the matching element type. Transport endpoints still compare full IDs and
structural formats exactly.

Each schema defines the scientific quantity, physical unit, coordinate and axis
meanings, logical ordering, validity rules, and compatibility policy. It does
not redefine the generic ndarray element type, shape, layout, or rate fields
owned by the graph host. Identifiers are opaque at runtime: PipeWire and graph
endpoints compare the complete identifier exactly and do not parse scientific
meaning from it. See the [architecture guide](../docs/architecture.md) for the
boundary between scientific schemas and graph transport fields.

## Shack–Hartmann summary packet

`pipewireao/shack-hartmann-summary-f32/1` is a CPU diagnostic packet of eight
Float32 values. Its inputs are aligned `slopes[n, 2]`, `flux[n]`, and
`validity[n]` from the same complete measurement frame. Output sample metadata
follows `slopes`.

Preparation requires 1 ≤ n ≤ 2²⁴ so both count fields are exact Float32 integers.

| Zero-based index | Quantity | Unit |
| --- | --- | --- |
| 0 | Total subaperture count | Count |
| 1 | Valid subaperture count | Count |
| 2 | Mean valid flux | Input flux unit |
| 3 | X slope RMS | Input slope unit |
| 4 | Y slope RMS | Input slope unit |
| 5 | hypot(X RMS, Y RMS) | Input slope unit |
| 6 | Mean X slope | Input slope unit |
| 7 | Mean Y slope | Input slope unit |

RMS means √mean(slope²), without subtracting the mean. Invalid rows are ignored
and accumulation uses Float64. With no valid rows or any nonfinite flux or
slope in a validity-true row, indices 2–7 are `NaN`; indices 0–1 retain finite
counts. This is successful diagnostic completion. Slope statistics preserve
detector-coordinate units, including detector pixels for pixel-coordinate
measurements. Mean slopes are not calibrated physical tip, tilt, or focus.
The producer does not infer a scientific calibration or alter measurement
outputs.
