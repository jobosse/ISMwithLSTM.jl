using Flux

@testset "LSTM.jl" begin

    @test typeof(SetUpLSTM(2,3)) ==  typeof(Chain(RNN(2 => 3), Dense(3 => 1)))
    @test regroupData([[1, 2, 3], [9, 9, 9]]...) == [[1, 9], [2, 9], [3, 9]]
    @test length(periodicForcing((1957,2011))) == 20088
    @test periodicForcing((1957,2011))[100] ≈ 0.6325734927397513

    LSTM = SetUpLSTM(2,3)
    saveLSTM(LSTM,"testLSTM")
    LSTM2 = loadLSTM("testLSTM.bson")
    rm("testLSTM.bson")
    data = regroupData([[1, 1, 1], [9, 9, 9]]...)
    @test typeof([LSTM2(x) for x in data]) == Vector{Vector{Float32}}

    #=
    f(x,y) = x+y
    saveLoss(f,"testLoss.bson")
    f2 = loadLoss("testLoss")
    rm("testLoss.bson")
    @test f2(2,3) ≈ 5
    =#
end