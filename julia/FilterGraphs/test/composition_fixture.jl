module CompositionFixture
using Test, FilterGraphAlgorithms, FilterGraphs
const FG = FilterGraphs
const FGA = FilterGraphAlgorithms

@algorithm CompositionGateF32 begin
    label = "composition-test-gate-f32"
    failure = SccPixelResidualFailure
    @configuration CompositionGateF32Configuration begin
        extent::Int => graph_rebuild
    end
    ports(configuration) = (
        input(:input,Float32,(configuration.extent,),FGA.CONTROLLER_RESIDUAL_ERROR_V1),
        output(:output,Float32,(configuration.extent,),FGA.CONTROLLER_RESIDUAL_ERROR_V1;metadata_from=:input),
        output(:validity,Bool,(1,),"pipewireao/center-of-gravity-validity-bool/1";metadata_from=:input))
    prepare(configuration) = nothing
    process!(plan) = begin
        input[1] == -99f0 && return SccPixelResidualIncompletePair
        copyto!(output,input)
        validity[1] = input[1] >= 0f0
        nothing
    end
end

function composition_fixture(directory; captured=false)
    for name in ("a","b")
        if captured
            write(joinpath(directory,name*".conf"),"""
            filter.graph = {
              nodes = [{name = $(name)-integrator type = ndarray label = closed-loop-integrator-f32
                config = {state_length = 1} props = {gain = 1.0 pole = 1.0 anti-windup-gain = 1.0}}]
              inputs = [{name = $(name)-image endpoint = $(name)-integrator:residual-error}
                        {name = $(name)-feedback endpoint = $(name)-integrator:constraint-feedback}]
              outputs = [{name = $(name)-correction endpoint = $(name)-integrator:correction}]
            }
            """)
            continue
        end
        gated = name == "b"
        label = gated ? "validity-gated-closed-loop-integrator-f32" : "closed-loop-integrator-f32"
        validity_config = gated ? "validity_length=1 validity_schema=\"pipewireao/center-of-gravity-validity-bool/1\"" : ""
        validity_link = gated ? "{output=b-gate:validity input=b-integrator:measurement-validity}" : ""
        offset_node = gated ? "" : "{name=a-reference type=ndarray label=vector-offset-f32 config={extent=1 schema=\"$(FGA.CONTROLLER_RESIDUAL_ERROR_V1)\"}}"
        residual_links = gated ? "{output=b-gate:output input=b-integrator:residual-error}" :
            "{output=a-gate:output input=a-reference:requested} {output=a-reference:output input=a-integrator:residual-error}"
        parameter = gated ? "" : "{name=a-reference endpoint=a-reference:offset}"
        write(joinpath(directory,name*".conf"),"""
        filter.graph = {
          nodes = [
            {name = $(name)-gate type = ndarray label = composition-test-gate-f32 config = {extent = 1}}
            $offset_node
            {name = $(name)-integrator type = ndarray label = $label
             config = {state_length = 1 $validity_config} props = {gain = 1.0 pole = 1.0 anti-windup-gain = 1.0}}
          ]
          links = [$residual_links $validity_link]
          inputs = [{name = $(name)-image endpoint = $(name)-gate:input}
                    {name = $(name)-feedback endpoint = $(name)-integrator:constraint-feedback} $parameter]
          outputs = [{name = $(name)-correction endpoint = $(name)-integrator:correction}]
        }
        """)
    end
    write(joinpath(directory,"shared.conf"),"""
    filter.graph = {
      nodes = [
        {name = sum type = ndarray label = vector-linear-combination-f32
         config = {extent = 1 left_scale = 1.0 right_scale = 1.0 schema = "$(FGA.CONTROLLER_COMMAND_V1)"}}
        {name = pdm-project type = ndarray label = dense-vector-reconstructor-f32
         config = {input_count = 1 output_count = 1 initial_reconstructor = [1]
          input_schema = "$(FGA.CONTROLLER_COMMAND_V1)" output_schema = "$(FGA.REQUESTED_PDM_COMMAND_V1)"
          reconstructor_schema = test.composition.controller-to-pdm/1}}
        {name = limit type = ndarray label = pdm-command-f32
         config = {actuator_count = 1 command_limits = [-0.5,0.5] slew_limit = null initial_current = null}}
        {name = feedback-to-controller type = ndarray label = dense-vector-reconstructor-f32
         config = {input_count = 1 output_count = 1 initial_reconstructor = [1]
          input_schema = "$(FGA.PDM_CONSTRAINT_FEEDBACK_V1)" output_schema = "$(FGA.CONTROLLER_CONSTRAINT_FEEDBACK_V1)"
          reconstructor_schema = test.composition.pdm-feedback-to-controller/1}}
        {name = a-return type = ndarray label = vector-linear-combination-f32
         config = {extent = 1 left_scale = 0.5 right_scale = 0.0 schema = "$(FGA.CONTROLLER_CONSTRAINT_FEEDBACK_V1)"}}
        {name = b-return type = ndarray label = vector-linear-combination-f32
         config = {extent = 1 left_scale = 0.5 right_scale = 0.0 schema = "$(FGA.CONTROLLER_CONSTRAINT_FEEDBACK_V1)"}}
      ]
      links = [{output = sum:output input = pdm-project:ordered-input}
       {output = pdm-project:output input = limit:requested}
       {output = limit:constraint-feedback input = feedback-to-controller:ordered-input}
       {output = feedback-to-controller:output input = a-return:left}
       {output = feedback-to-controller:output input = a-return:right}
       {output = feedback-to-controller:output input = b-return:left}
       {output = feedback-to-controller:output input = b-return:right}]
      inputs = [{name = a-command endpoint = sum:left} {name = b-command endpoint = sum:right}]
      outputs = [{name = command endpoint = limit:demanded}
       {name = requested endpoint = sum:output}
       {name = a-returned endpoint = a-return:output} {name = b-returned endpoint = b-return:output}]
    }
    """)
    descriptor = joinpath(directory,"composition.conf")
    write(descriptor,"""
    filter.composition = {
      branches = [
        {name = a graph = a.conf input = a-image correction = a-correction feedback = a-feedback
         shared-input = a-command shared-feedback = a-returned}
        {name = b graph = b.conf input = b-image correction = b-correction feedback = b-feedback
         shared-input = b-command shared-feedback = b-returned}
      ]
      shared = shared.conf
    }
    """)
    return descriptor
end


end
