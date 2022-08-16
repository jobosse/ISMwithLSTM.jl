using Flux
using Statistics
using BSON: @save
using BSON: @load

include("ReadInputData.jl")
include("ProximityFunctions.jl")



"""
    SetUpLSTM(in_size,C_size)

Sets up an LSTM neural network (for reference check out [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
where the input dimension in each recurrent step is given by `in_size` and the dimension of the cell state by `C_size`.
The dimension of the state which the acticle calls 'h' is the same as the dimension of the cell state, so the final output is
computed by a single dense layer mapping 'h' to a scalar value (i.e. the estimate for the proximity function at the given time).


"""
function SetUpLSTM(in_size,C_size)
    return Chain(RNN(in_size => C_size), Dense(C_size => 1))
end


"""
    trainLSTM!(LSTM,
    train_period::Tuple=(1948,1980),
    test_period::Tuple=(1981,2010);
    Tr = 366+365,
    epochs = 100,
    λ1 = 0.,
    λ2 = 0.,
    learning_rate = 1e-3)

Trains a given LSTM network on training data and hyperparameters specified by the arguments of the function

# Arguments
- `LSTM`: The LSTM to be trained 
- `train_period::Tuple=(1948,1980)`: Start and end year of training
- `test_period::Tuple=(1981,2010)`: Start and end year of testing
- `Tr = 366+365`: Transient time in days. The model will not be trained on the first `Tr` number of days of the training set
- `epochs = 100`: Number of maximum epochs (early stopping possible if loss diverges or loss reaches a platteu)
- `λ1 = 0.`: Weight of the L1 regularisation
- `λ2 = 0.`: Weight of the L2 regularisation
- `learning_rate = 1e-3`: Initial learning rate. It reduces by a factor of two every 30 epochs.

# Returns
- Trained LSTM
- Array of train losses (one loss value per epoch)
- Array of test losses (one loss value per epoch)
"""
function trainLSTM!(LSTM,
    train_period::Tuple=(1948,1980),
    test_period::Tuple=(1981,2010);
    Tr = 366+365,
    epochs = 100,
    λ1 = 0.,
    λ2 = 0.,
    learning_rate = 1e-3)
#=
    ΔTT_transient = ...
    ΔTT_train = ...
    ΔTT_test = ...
    prox_train = ...
    prox_test = ...
=#
    # Define loss function with regularisation
    L1(θ) = sum(x -> sum(abs, x), θ)
    L2(θ) = sum(x -> sum(abs2, x), θ)
    loss(ΔTT,prox) = (sum(Flux.mse(LSTM(xi), yi) for (xi, yi) in zip(ΔTT,prox)) 
        + λ1*L1(Flux.params(LSTM)) + λ2*L2(Flux.params(LSTM)))

    opt= Adam(learning_rate)

    train_loss = []
    test_loss = []

    println("starting training...")
    for epoch in 1:epochs     
        # Run the model for the transient period not included in the loss function
        Flux.reset!(LSTM)
        [LSTM(x) for x in ΔTT_transient]

        train_data = zip(ΔTT_train,prox_train)
        Flux.train!(loss, Flux.params(mlp), train_data, opt)
        if (i_e % 30) == 0  # reduce the learning rate every 30 epochs
            opt.eta /= 2
            println("reduced learning rate to ", opt.eta)
        end
         
        # record losses
        push!(train_loss, loss(ΔTT_train,prox_train))
        push!(test_loss, loss(ΔTT_test,prox_test))
        println("epoch: ", epoch, ",   train loss: ", train_loss[end], ",   test loss: ", test_loss[end])


        if epoch < 10
            if train_loss > 1e7
                println("train loss ist diverging")
                break
            end
            if std(test_loss) < 1
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
"""
function saveLSTM(LSTM,path::String)
    last(path,5) == ".bson" && path = chop(path, tail=5)
    @save string(path,".bson") LSTM
end

"""
    function loadLSTM(LSTM,path::String)

Returns the  LSTM saved at the given location.
"""
function loadLSTM(path::String)
    last(path,5) == ".bson" && path = chop(path, tail=5)
    @load string(path,".bson") LSTM
    return LSTM
end