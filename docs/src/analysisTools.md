```@docs
RunLSTM(LSTM, paths_to_data::Vector{String},end_year::Int)
```
```@docs
CalculateLoss(loss, LSTM, paths_to_data::Vector{String},pr::ProxFct,run_period::Tuple{Int64,Int64})
```
```@docs
OnsetDayPrediction(LSTM, paths_to_data::Vector{String}, yr::Int; t_1::Int = 60, t_2::Int = 70)
```
```@docs
PlotProximity(LSTM, paths_to_data::Vector{String}, path_to_zero_crossings::String, time_period::Tuple{Int64,Int64})
```
```@docs
PlotOnsetComparsion(LSTM, paths_to_data::Vector{String}, 
        path_to_zero_crossings::String, 
        time_period::Tuple{Int64,Int64};
        t_1= 60::Int, 
        t_2=70::Int, 
        fig_name = "OnsetComparison_$(time_period)[1]_$(time_period)[2].pdf"::String)
```
```@docs
function PlotLeadTimeAnalysis(LSTM, 
    paths_to_data::Vector{String}, 
    path_to_zero_crossings::String,
    save_directory::String; 
    lead_time_range = (60,110),
    time_period = (1981,2020)::Tuple{Int64,Int64})
```