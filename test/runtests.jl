using ISMwithLSTM
using Test

@testset "ISMwithLSTM.jl" begin
    include("ProximityFunctions.jl")
    include("ReadInputData.jl")
    include("LSTM.jl")
end
 