using Flux
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

function CalculateLoss(loss, LSTM, paths_to_data::Vector{String},path_to_prox::String,run_period::Tuple{Int64,Int64})
    # Reset LSTM and run it until start period
    Flux.reset!(LSTM)
    if run_period[1] > 1948
        transient_period = (1948,run_period[1]-1)
        transient_data = regroupData([LoadAnnualData(transient_period,path)[:,2] for path in paths_to_data]...,periodicForcing(transient_period))
        [LSTM(x) for x in transient_data]
    end
    # Calculate loss for run_period
    data = regroupData([LoadAnnualData(run_period,path)[:,2] for path in paths_to_data]...,periodicForcing(run_period))
    prox = [Vector{Float32}([data]) for data in LoadAnnualData(run_period,path_to_prox)[:,2]]
    loss_value = loss(data,prox,LSTM)
    println("Validation loss for years $(run_period[1])-$(run_period[2]): $loss_value")
    return loss_value
end