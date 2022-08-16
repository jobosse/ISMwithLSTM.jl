using Test
using DelimitedFiles

@testset "ReadInputData.jl" begin
    a = readdlm("TestData/deltaTT.txt")
    @test all(LoadAnnualData((1948,1949), "TestData/deltaTT.txt") .≈ a[:,1:2])
end