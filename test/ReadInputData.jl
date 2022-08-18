using Test
using DelimitedFiles

@testset "ReadInputData.jl" begin
    a = readdlm("TestData/test_dTT.txt")
    @test all(LoadAnnualData((1948,1949), "TestData/test_dTT.txt") .≈ a[:,1:2])
end