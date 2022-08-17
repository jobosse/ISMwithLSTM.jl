
using Flux
using Statistics
using BSON: @save
using BSON: @load
using Logging

include("ReadInputData.jl")
include("ProximityFunctions.jl")



"""
    SetUpLSTM(in_size,C_size)

Sets up an LSTM neural network (for reference check out [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
where the input dimension in each recurrent step is given by `in_size` and the dimension of the cell state by `C_size`.
The dimension of the state which the acticle calls 'h' is the same as the dimension of the cell state, so the final output is
computed by a single dense layer mapping 'h' to a scalar value (i.e. the estimate for the proximity function at the given time).
"""
function SetUpLSTM(in_size::Int64,C_size::Int64)
    return Chain(RNN(in_size => C_size), Dense(C_size => 1))
end


# Returns array for periodic forcing wich is included in the training data.
function periodicForcing(time_period::Tuple{Int64, Int64})
    start_yr, end_yr = time_period
    lap_yrs = 0
    for i in start_yr:end_yr
        i % 4 == 0 && (lap_yrs += 1)
    end
    offset = (start_yr % 4 == 0 ? 0 : (4 - (start_yr % 4))) * 0.25
    days = [i for i in 1:(365*(end_yr-start_yr+1)+lap_yrs)] .+ offset
    return cos.(2*pi*(days.-152.25)./365.25)
end


# Tool to turn the data into a form suitable for the LSTM
regroupData(data...) = [Vector{Float32}(collect(x)) for x in zip(data...)]


"""
    function trainLSTM(path_to_prox::String,
        paths_to_data::Vector{String},
        train_period::Tuple{Int64, Int64}=(1948,1980),
        test_period::Tuple{Int64, Int64}=(1981,2010);
        C_dim::Int64 = 5,
        Tr = 366+365::Int64,
        epochs = 100::Int64,
        λ1 = 0.::Float64,
        λ2 = 0.::Float64,
        learning_rate = 1e-3::Float64)

Trains a given LSTM network on training data and hyperparameters specified by the arguments of the function

# Arguments
- `path_to_prox::String`: Path to the proximity data
- `paths_to_data::Vector{String}`: Array of strings describing the paths to the data which should be used for training
- `train_period::Tuple{Int64, Int64}=(1948,1980)`: Start and end year of training
- `test_period::Tuple{Int64, Int64}=(1981,2010)`: Start and end year of testing
- `C_dim::Int64 = 5`: Dimension of the cell state of the LSTM (for reference check out [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
- `Tr::Int64 = 366+365`: Transient time in days. The model will not be trained on the first `Tr` number of days of the training set
- `epochs::Int64 = 100`: Number of maximum epochs (early stopping possible if loss diverges or loss reaches a platteu)
- `λ1::Float64 = 0.`: Weight of the L1 regularisation
- `λ2::Float64 = 0.`: Weight of the L2 regularisation
- `learning_rate::Float64 = 1e-3`: Initial learning rate. It reduces by a factor of two every 30 epochs.

# Returns
- Trained LSTM
- Array of train losses (one loss value per epoch)
- Array of test losses (one loss value per epoch)
"""
function trainLSTM(path_to_prox::String,
    paths_to_data::Vector{String},
    train_period::Tuple{Int64, Int64}=(1948,1980),
    test_period::Tuple{Int64, Int64}=(1981,2010);
    C_dim::Int64 = 5,
    Tr = 365+365::Int64,
    epochs = 100::Int64,
    λ1 = 0.::Float64,
    λ2 = 0.::Float64,
    learning_rate = 1e-3::Float64)

    # create the datasets
    transient_data = regroupData([LoadAnnualData(train_period,path)[:,2] for path in paths_to_data]...,periodicForcing(train_period))[1:Tr]
    train_data = regroupData([LoadAnnualData(train_period,path)[:,2] for path in paths_to_data]...,periodicForcing(train_period))[Tr+1:end]
    test_data = regroupData([LoadAnnualData(test_period,path)[:,2] for path in paths_to_data]...,periodicForcing(test_period))
    prox_train = [Vector{Float32}([data]) for data in LoadAnnualData(train_period,path_to_prox)[:,2]][Tr+1:end]
    prox_test = [Vector{Float32}([data]) for data in LoadAnnualData(test_period,path_to_prox)[:,2]]

    # Create model
    LSTM = SetUpLSTM(length(paths_to_data)+1,C_dim)

    # Define loss function with regularisation
    L1(θ) = sum(x -> sum(abs, x), θ)
    L2(θ) = sum(x -> sum(abs2, x), θ)
    loss(data,prox) = (sum([Flux.mse(LSTM(xi), yi) for (xi, yi) in zip(data,prox)]) 
        + λ1*L1(Flux.params(LSTM)) + λ2*L2(Flux.params(LSTM)))

    opt= Adam(learning_rate)

    train_loss = []
    test_loss = []
    println("starting training...")
    for epoch in 1:epochs     
        # Run the model for the transient period not included in the loss function
        Flux.reset!(LSTM)
        [LSTM(x) for x in transient_data]

        data = zip([train_data],[prox_train])
        Flux.train!(loss, Flux.params(LSTM), data, opt)
        if (epoch % 30) == 0  # reduce the learning rate every 30 epochs
            opt.eta /= 2
            println("reduced learning rate to ", opt.eta)
            println("")
        end
         
        # record losses
        Flux.reset!(LSTM)
        [LSTM(x) for x in transient_data]
        push!(train_loss, loss(train_data,prox_train))
        if train_period[2]+1 != test_period[1]
            
        end
        push!(test_loss, loss(test_data,prox_test))
        println("epoch: ", epoch)
        println("train loss: ", train_loss[end], ",   test loss: ", test_loss[end])
        println("")


        if epoch > 10
            if train_loss[end] > 1e7
                println("train loss ist diverging")
                break
            end
            if std(test_loss[end-5:end]) < 1
                println("early stopping du to platteu in test loss")
                break
            end
        end
    end
    return LSTM, test_loss, train_loss
end


"""
    function saveLSTM(LSTM,path::String)
    
Saves the  LSTM as a '.bson' file at the given location.

# Arguments
* `LSTM, returned from trainLSTM() `
* `path::String, e.g. "parameters/my_lstm.bson"`
"""
function saveLSTM(LSTM,path::String)
    last(path,5) == ".bson" && (path = chop(path, tail=5))
    @save string(path,".bson") LSTM
end


"""
    function loadLSTM(LSTM,path::String)

Returns the  LSTM saved at the given location.

# Arguments
* `path::String, e.g. "parameters/my_lstm.bson"`
"""
function loadLSTM(path::String)
    last(path,5) == ".bson" && (path = chop(path, tail=5))
    @load string(path,".bson") LSTM
    return LSTM
end