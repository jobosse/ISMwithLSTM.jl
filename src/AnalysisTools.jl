using CSV  

"""
Load in raw ΔTT data using CSV.jl package

# Arguments

* 'data_path::String'


# Returns
* CSV.File with columns : ["ΔTT", "yr", "day", "T_n","T_s"]
"""
function LoadData(data_path::String)
    data = CSV.File(data_path, header=["ΔTT", "yr", "day", "T_n","T_s"])
    return data
end
