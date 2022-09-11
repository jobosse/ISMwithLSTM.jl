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
PlotProximity(LSTM,pr::ProxFct, paths_to_data::Vector{String}, time_period::Tuple{Int64,Int64}, save_directory::String)
```
```@docs
 PlotOnsetComparsion(LSTM, 
        paths_to_data::Vector{String}, 
        pr::ProxFct,
        time_period::Tuple{Int64,Int64};
        t_1= 60::Int, 
        t_2=70::Int,
        save_directory::String,
        ylim = (120,190))
```
```@docs
WriteOutLeadTimeAnalysis(LSTM,  
    paths_to_data::Vector{String}, 
    pr::ProxFct,
    save_directory::String; 
    lead_time_range = (60,110),
    time_period = (1981,2020)::Tuple{Int64,Int64})
```
```@docs
PlotLeadTimeAnalysis( 
    path_to_data::String, 
    save_directory::String; 
    lead_time_range = (60,110),
    time_period = (1981,2020)::Tuple{Int64,Int64})
```