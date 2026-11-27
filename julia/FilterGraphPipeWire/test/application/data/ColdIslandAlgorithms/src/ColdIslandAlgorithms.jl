module ColdIslandAlgorithms

import FilterGraphAlgorithms

FilterGraphAlgorithms.@algorithm ScaleF32 begin
    label = "cold-island-scale-f32"

    @configuration ScaleConfiguration begin
        extent::Int => graph_rebuild
        gain::Float32 => construction
    end

    ports(configuration) = (
        input(:input, Float32, (configuration.extent,), "test.signal.f32/1"),
        output(:output, Float32, (configuration.extent,), "test.signal.f32/1";
            metadata_from=:input),
    )
    prepare(configuration) = configuration.gain
    process!(plan) = begin
        @. output = plan * input
        nothing
    end
end

algorithms() = (ScaleF32,)

export ScaleF32

end
