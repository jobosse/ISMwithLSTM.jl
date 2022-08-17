#include("../src/LSTM.jl")

@testset "AnalysisTools.jl" begin
    test_lstm = loadLSTM("TestData/test_lstm.bson")
    test_data_path = "TestData/deltaTT.txt"
    @test RunLSTM(test_lstm, [test_data_path],(1948,1949))[end,2] ≈ 0.318445 atol=0.0001
    
end