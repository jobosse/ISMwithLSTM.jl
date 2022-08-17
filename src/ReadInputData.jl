using DelimitedFiles
using Logging
include("HelperFunctions.jl")
include("ProximityFunctions.jl")

"""
    function WriteSignSwitchesIndices(data, save_to_path::String)
Writes out all zero-crossing of ΔTT in a txt file

# Arguments

* 'data::Matrix{Float64}', with columns: ["ΔTT", "yr", "day", "T_n","T_s"]
* `save_to_path::String`, path to save to.

"""
function WriteSignSwitchesIndices(data, save_to_path::String)
    index_sign = []
    for i in unique(data[:,1])
        index_year = findallIndex(x -> x==i,data[:,1])
        ΔTT_yearly = data[index_year,2]
        push!(index_sign,FindSignSwitch(ΔTT_yearly).+index_year[1].-1)  
    end
    index_sign = [(index_sign...)...] # Flatten AbstractArray

    outfile = save_to_path * "zero_crossings.txt"
    try 
        open(outfile, "w") do f
            for i in index_sign
                println(f, i)
            end
        end
        @info "zero_crossing.txt was created in \"$save_to_path\". \n It holds all zero-crossing-points of ΔTT."
    catch e
        @warn "Could not write zero_crossing.txt in  \"$save_to_path\". Does the directory exist?"
    end
end


"""
    function WriteProximityData(data_path::String, save_to_path::String = "", output_file_name::String = "proximity.txt"; ProximityFunction::Function = LinearProximityFunction)

Writes out the values of given Proximity Function for all data points into txt file. 
The columns of the txt file correspond to: ["year","ProximityFunctionValue"]

# Arguments
* `data_path::String, path to input data`
* `save_to_path::String = "", path to directory where the outputfile is saved.``
* `output_file_name::String, name of the outputfile`
# Keyword Arguments
* `ProximityFunction::Function = LinearProximityFunction``

"""
function WriteProximityData(data_path::String, save_to_path::String = "", output_file_name::String = "proximity.txt"; ProximityFunction::Function = LinearProximityFunction)
    if save_to_path == ""
        save_to_path = pwd() * "/"
    end

    if !(last(save_to_path) == '/')
        save_to_path = save_to_path * "/"
    end

    data = readdlm(data_path)
    # Create 'zero_crossings.txt' file which is needed for the ProximityFunction
    WriteSignSwitchesIndices(data, save_to_path)

    output = zeros((length(data[:,1]), 2))
    first_index = 157 # This is the first index where ΔTT crosses 0
    for i in 1:(length(data[:,1]))
        output[i,1] = Int64(data[i,1]) # Adding the years
        if i < first_index
            output[i,2] = 0 # Adding dummy values
        else
            output[i,2] = LinearProximityFunction(i, save_to_path)
       end
    end
    outfile =  save_to_path * output_file_name
    open(outfile, "w") do io
        writedlm(io, output)
    end
    @info "$output_file_name was created in \"$save_to_path \". \n It holds all values of the given ProximityFunction."
end

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
