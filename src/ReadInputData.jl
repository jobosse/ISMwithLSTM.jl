using DelimitedFiles
using Logging
include("HelperFunctions.jl")
include("ProximityFunctions.jl")


"""
    function LoadAnnualData(time_period::Tuple{Real,Real}, data_path::String)

Returns array with column: [year, data] for the annual time period given.

# Arguments
* `time_period::Tuple{Real,Real}`
* `data_path::String`

# Returns
* `Matrix{Float64}`
"""
function LoadAnnualData(time_period::Tuple{Real,Real}, data_path::String)
    data = readdlm(data_path)
    time_period = Int.(time_period)
    t_1 = time_period[1]
    t_2 = time_period[2]
    t_1_index = findallIndex(x-> x==t_1, Int.(data[:,1]))
    t_2_index = findallIndex(x-> x==t_2, Int.(data[:,1]))
    t_1_index = findmin(t_1_index)[1]
    t_2_index = findmax(t_2_index)[1]
    time_period_index = t_1_index:1:t_2_index
    a = hcat(Int.(data[:,1][time_period_index]),data[:,2][time_period_index])
    return a
end
