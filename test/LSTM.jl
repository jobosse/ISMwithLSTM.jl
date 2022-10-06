using Flux

@testset "LSTM.jl" begin

    @test typeof(SetUpLSTM(2,3)) ==  typeof(Chain(LSTM(2 => 3), Dense(3 => 1)))
    @test regroupData([[1, 2, 3], [9, 9, 9]]...) == [[1, 9], [2, 9], [3, 9]]
    @test length(periodicForcing((1957,2011))) == 20088
    @test periodicForcing((1957,2011))[100] ≈ 0.6325734927397513

    LSTM = SetUpLSTM(2,3)
    saveLSTM(LSTM,"testLSTM")
    LSTM2 = loadLSTM("testLSTM.bson")
    rm("testLSTM.bson")
    data = regroupData([[1, 1, 1], [9, 9, 9]]...)
    @test typeof([LSTM2(x) for x in data]) == Vector{Vector{Float32}}

    pr = ProxFct("TestData/test_dTT.txt")
    lstm, test_loss, train_loss, helper_loss = trainLSTM(pr,["TestData/test_dTT.txt"],(1948,1948),(1949,1949),Tr=2,epochs=10,reduce_learning_rate=5)
    @test typeof([lstm(x) for x in data]) == Vector{Vector{Float32}}
    @test length(test_loss) == 10
    @test length(train_loss) == 10
    @test typeof(helper_loss(data,pr(1948)[1][1:3],lstm)) == Float64
end