using Flux

@testset "AnalysisTools.jl" begin

    lstm = SetUpLSTM(2,3)
    lstm2 = RunLSTM(lstm,["TestData/test_dTT.txt"],1949)
    data = regroupData([[1, 1, 1], [9, 9, 9]]...)
    @test typeof([lstm2(x) for x in data]) == Vector{Vector{Float32}}

    loss(data,prox,LSTM) = 1/(length(data[:,1]))*(sum([Flux.mse(LSTM(xi), yi) for (xi, yi) in zip(data,prox)]))
    pr = ProxFct("TestData/test_dTT.txt")
    @test typeof(CalculateLoss(loss, lstm, ["TestData/test_dTT.txt"],pr::ProxFct,(1948,1949))) == Float64
    @test typeof(OnsetDayPrediction(lstm, ["TestData/test_dTT.txt"], 1949)) == Float32
    #test_lstm = loadLSTM("TestData/test_lstm.bson")
    #test_data_path = "TestData/deltaTT.txt"
    #@test RunLSTM(test_lstm, [test_data_path],(1948,1949))[end,2] ≈ 0.318445 atol=0.0001
    
end