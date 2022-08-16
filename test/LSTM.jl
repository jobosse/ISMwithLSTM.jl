using Flux

@testset "LSTM.jl" begin

    @test typeof(SetUpLSTM(2,3)) ==  typeof(Chain(RNN(2 => 3), Dense(3 => 1)))
    @test regroupData([[1, 2, 3], [9, 9, 9]]...) == [[1, 9], [2, 9], [3, 9]]
    @test length(periodicForcing((1957,2011))) == 20088
    @test periodicForcing((1957,2011))[100] ≈ 0.6325734927397513

end