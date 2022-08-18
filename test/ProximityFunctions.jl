using Test

@testset "ProximityFunctions.jl" begin

    @test(LinearProximityFunction(170, "TestData/zero_crossings.txt") ≈ 0.8968253968253969)
end