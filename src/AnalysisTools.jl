include("LSTM.jl")
include("ReadInputData.jl")

"""
    function RunLSTM(LSTM, paths_to_data::Vector{String},run_period::Tuple{Int64, Int64})

Runs given LSTM on the data over the given period

# Arguments
- `LSTM`
- `paths_to_data::Vector{String}`: Array of strings describing the paths to the data which should be used for training
- `run_period::Tuple{Int64, Int64}`: Has to be of the form (start_year,end_year)
"""
function RunLSTM(LSTM, paths_to_data::Vector{String},run_period::Tuple{Int64, Int64})
    input_data = regroupData([LoadAnnualData(run_period,path)[:,2] for path in paths_to_data]...,periodicForcing(run_period))
    years = LoadAnnualData(run_period,paths_to_data[1])[:,1]
    result = [LSTM(x)[1] for x in input_data]
    result = hcat(years,result)
    return result
end