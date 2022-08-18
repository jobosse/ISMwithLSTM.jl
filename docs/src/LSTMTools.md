```@docs
SetUpLSTM(in_size::Int64,C_size::Int64)
```

```@docs
trainLSTM(pr::ProxFct,
    paths_to_data::Vector{String},
    train_period::Tuple{Int64, Int64}=(1948,1980),
    test_period::Tuple{Int64, Int64}=(1981,2010);
    C_dim::Int64 = 5,
    Tr::Int64 = 365+365,
    epochs::Int64 = 100,
    λ1::Float64 = 0.,
    λ2::Float64 = 0.,
    learning_rate::Float64 = 1e-2,
    reduce_learning_rate::Int64 = 20)

```

```@docs
saveLSTM(LSTM,path::String)
```

```@docs
loadLSTM(path::String)
```

```@docs
saveFct(fct,path::String)
```

```@docs
loadFct(path::String)
```

```@docs
ProxFct(path_to_data::String)
```
