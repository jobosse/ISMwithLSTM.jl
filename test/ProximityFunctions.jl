using Test

@testset "ProximityFunctions.jl" begin

    a = hcat([1948,1949],[157,151])

    pr = ProxFct("TestData/test_dTT.txt")
    @test pr.ProxData[300,2] ≈ 0.07264957264957266
    @test all(size(pr.ProxData) == (731, 2))
    @test all(pr.onsets == a)
    @test pr(1949)[1][2] ≈ 0.3632478632478633
    @test all(pr((1948,1949))[2]==[157,151])
end