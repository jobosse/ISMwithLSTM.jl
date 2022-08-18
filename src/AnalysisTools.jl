using Flux
using CurveFit
using Plots
using Statistics
include("LSTM.jl")
include("ReadInputData.jl")
include("HelperFunctions.jl")
include("ProximityFunctions.jl")


"""
    function RunLSTM(LSTM, paths_to_data::Vector{String},run_period::Tuple{Int64, Int64})

Runs given LSTM on the data over the given period

# Arguments
- `LSTM`
- `paths_to_data::Vector{String}`: Array of strings describing the paths to the data which should be used for training
- `run_period::Tuple{Int64, Int64}`: Has to be of the form (start_year,end_year)
"""
function RunLSTM(LSTM, paths_to_data::Vector{String},end_year::Int)
    # Reset LSTM and run it until start period
    Flux.reset!(LSTM)
    if end_year > 1947
        run_period = (1948,end_year)
        input_data = regroupData([LoadAnnualData(run_period,path)[:,2] for path in paths_to_data]...,periodicForcing(run_period))
        [LSTM(x) for x in input_data]
    end
    return LSTM
end

"""
    function CalculateLoss(loss, LSTM, paths_to_data::Vector{String},path_to_prox::String,run_period::Tuple{Int64,Int64})

Calculates loss over the given run_period.

# Arguments
- `loss` : The loss function which is used 
- `LSTM`
- `paths_to_data::Vector{String}`
- `path_to_prox::String`
- `run_period::Tuple{Int64,Int64}`

# Returns
- loss_value::Float64
"""
function CalculateLoss(loss, LSTM, paths_to_data::Vector{String},path_to_prox::String,run_period::Tuple{Int64,Int64})
    # Reset LSTM and run it until start period
    LSTM = RunLSTM(LSTM, paths_to_data, run_period[1]-1)
    # Calculate loss for run_period
    data = regroupData([LoadAnnualData(run_period,path)[:,2] for path in paths_to_data]...,periodicForcing(run_period))
    prox = [Vector{Float32}([data]) for data in LoadAnnualData(run_period,path_to_prox)[:,2]]
    loss_value = loss(data,prox,LSTM)
    println("Validation loss for years $(run_period[1])-$(run_period[2]): $loss_value")
    return loss_value
end

"""
    function OnsetDayPrediction(LSTM, paths_to_data::Vector{String}, yr::Int, t_1 = 60::Int, t_2 = 70::Int)

# Arguments
- `LSTM`
- `paths_to_data::Vector{String}`
- `yr::Int`: year to predict the onset for
- `t_1 = 60::Int`: corresponds to the number of days before January 1st of the prediction year
- `t_2 = 60::Int`: correpsonds to the number of days after January 1st of the prediction year

# Returns
- `ISM Onset Day`
"""
function OnsetDayPrediction(LSTM, paths_to_data::Vector{String}, yr::Int; t_1 = 60::Int, t_2 = 70::Int)
    Flux.reset!(LSTM)
    run_period = (1948,yr)
    data = regroupData([LoadAnnualData(run_period,path)[:,2] for path in paths_to_data]...,periodicForcing(run_period))
    yrs = Int.(LoadAnnualData(run_period,paths_to_data[1])[:,1])
    index_jan_1 = findallIndex(x-> x== yr,yrs)[1]
    lstm_values = [LSTM(x)[1] for x in data][(index_jan_1-t_1):(index_jan_1+t_2)]
    a,b = linear_fit(collect(-t_1:1:t_2),lstm_values) # f(x) = a*x + b
    return (1-a)/b
end

"""
    function PlotProximity(LSTM, paths_to_data::Vector{String}, path_to_zero_crossings::String, time_period::Tuple{Int64,Int64})

Plots the real Proximity Function vs. the learned one on the given time_period

# Arguments 
- `LSTM`
- `paths_to_data::Vector{String}`
- `path_to_zero_crossings::String`
- `time_period::Tuple{Int64,Int64}`
"""
function PlotProximity(LSTM, paths_to_data::Vector{String}, path_to_zero_crossings::String, time_period::Tuple{Int64,Int64})
    # Reset LSTM 
    Flux.reset!(LSTM)
    full_period = (1948,2022)
    data = regroupData([LoadAnnualData(full_period,path)[:,2] for path in paths_to_data]...,periodicForcing(full_period))
    yrs =  LoadAnnualData(full_period,paths_to_data[1])[:,1]
    lstm_values = [LSTM(x)[1] for x in data]
    proximity_values = [LinearProximityFunction(x, path_to_zero_crossings) for x in 1:length(data)]
    start_index = findallIndex(x-> x == Int(time_period[1]),yrs)[1]
    end_index = findallIndex(x-> x == Int(time_period[2]),yrs)[end]
    plot(proximity_values[start_index:end_index])
    plot!(lstm_values[start_index:end_index])
end

"""
"""
function PlotOnsetComparsion(LSTM, paths_to_data::Vector{String}, 
        path_to_zero_crossings::String, 
        time_period::Tuple{Int64,Int64};
        t_1= 60::Int, 
        t_2=70::Int, 
        fig_name = "OnsetComparison_$(time_period)[1]_$(time_period)[2].pdf"::String)

    pred_on_set = [OnsetDayPrediction(LSTM, paths_to_data, yr,t_1,t_2) for yr in time_period]
    ΔTT_on_set = ProxFct(time_period)[2]# @daniel lies mal hier bitte echten onsets rein für die jahre in time_period
    x_axis = collect(time_period[1]:1:time_period[2])    
    y_data = hcat(pred_on_set, ΔTT_on_set)
    x_ticks = ["$i" for i in x_axis]
    plot(y_data, 
        label = ["Actual OD" "Predicted OD"],
        legend=:bottomright,
        color = [:black :orange], 
        line = (:dot, 2), 
        marker = ([:hex :d], 5, 0.9),
        ylim = (130,190),
        xticks = (1:length(x_axis), x_ticks))
    ylabel!("Onset Date (OD)")
    savefig(fig_name)
end

"""
days between the issuing of the forecast at t2 and June 2nd (the average onset date of data)
"""
function PlotLeadTimeAnalysis(LSTM, 
    paths_to_data::Vector{String}, 
    path_to_zero_crossings::String,
    save_directory::String; 
    lead_time_range = (60,110),
    time_period = (1981,2020)::Tuple{Int64,Int64})

    t_2_range =  (153 .- lead_time_range)[1]:1:(153 .- lead_time_range)[end]
    std_vec = []
    cor_vec = []
    ΔTT_on_set = ProxFct(time_period)[2]# @daniel lies mal hier bitte echten onsets rein für die jahre in time_period
        #real onset dates

    for t_2 in t_2_range
        onset_days = [OnsetDayPrediction(LSTM, paths_to_data, yr,t_2=t_2) for yr in time_period]
        push!(std_vec, std(onset_days, corrected = false))
        push!(cor_vec, cor(onset_days,ΔTT_on_set))
    end

    x_ticks = ["$i" for i in t_2_range]
    plot(std_vec, 
        label = ["Prediction years. \n $(time_period[1])-$(time_period[2])"],
        legend=:bottomright,
        color = [:orange], 
        line = (:dot, 2), 
        marker = ([:hex :d], 5, 0.9),
        xticks = (1:length(x_ticks), x_ticks))
    ylabel!("RMSE (day)")
    xlabel!("Average lead time (day)")
    hline!([5.97], color=:orange, linestyle=:dash)
    savefig(save_directory * "Std_LeadTime_$(time_period)[1]_$(time_period)[2].pdf")

    plot(cor_vec, 
    label = ["Prediction years. \n $(time_period[1])-$(time_period[2])"],
    legend=:bottomright,
    color = [:orange], 
    line = (:dot, 2), 
    marker = ([:hex :d], 5, 0.9),
    xticks = (1:length(x_ticks), x_ticks))
    ylabel!("Correlation")
    xlabel!("Average lead time (day)")
    savefig(save_directory * "Cor_LeadTime_$(time_period)[1]_$(time_period)[2].pdf")

end 
