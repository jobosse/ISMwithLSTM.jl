var documenterSearchIndex = {"docs":
[{"location":"analysisTools/","page":"Analysis Tools","title":"Analysis Tools","text":"RunLSTM(LSTM, paths_to_data::Vector{String},end_year::Int)","category":"page"},{"location":"analysisTools/#ISMwithLSTM.RunLSTM-Tuple{Any, Vector{String}, Int64}","page":"Analysis Tools","title":"ISMwithLSTM.RunLSTM","text":"function RunLSTM(LSTM, paths_to_data::Vector{String},end_year::Int)\n\nRuns given LSTM on the data over the given period\n\nArguments\n\nLSTM\npaths_to_data::Vector{String}: Array of strings describing the paths to the data which should be used for training\nend_year::Int: Year until which the LSTM is run\n\n\n\n\n\n","category":"method"},{"location":"analysisTools/","page":"Analysis Tools","title":"Analysis Tools","text":"CalculateLoss(loss, LSTM, paths_to_data::Vector{String},pr::ProxFct,run_period::Tuple{Int64,Int64})","category":"page"},{"location":"analysisTools/#ISMwithLSTM.CalculateLoss-Tuple{Any, Any, Vector{String}, ProxFct, Tuple{Int64, Int64}}","page":"Analysis Tools","title":"ISMwithLSTM.CalculateLoss","text":"function CalculateLoss(loss, LSTM, paths_to_data::Vector{String},pr::ProxFct,run_period::Tuple{Int64,Int64})\n\nCalculates loss over the given run_period.\n\nArguments\n\nloss: The loss function which is used \nLSTM\npaths_to_data::Vector{String}\npr::ProxFct\nrun_period::Tuple{Int64,Int64}\n\nReturns\n\nloss_value::Float64\n\n\n\n\n\n","category":"method"},{"location":"analysisTools/","page":"Analysis Tools","title":"Analysis Tools","text":"OnsetDayPrediction(LSTM, paths_to_data::Vector{String}, yr::Int; t_1::Int = 60, t_2::Int = 70)","category":"page"},{"location":"analysisTools/#ISMwithLSTM.OnsetDayPrediction-Tuple{Any, Vector{String}, Int64}","page":"Analysis Tools","title":"ISMwithLSTM.OnsetDayPrediction","text":"function OnsetDayPrediction(LSTM, paths_to_data::Vector{String}, yr::Int; t_1::Int = 60, t_2::Int = 70)\n\nArguments\n\nLSTM\npaths_to_data::Vector{String}\nyr::Int: year to predict the onset for\n\nKeyword Arguments\n\nt_1::Int = 60: corresponds to the number of days before January 1st of the prediction year\nt_2::Int = 70: correpsonds to the number of days after January 1st of the prediction year\n\nReturns\n\nISM Onset Day\n\n\n\n\n\n","category":"method"},{"location":"analysisTools/","page":"Analysis Tools","title":"Analysis Tools","text":"PlotProximity(LSTM,pr::ProxFct, paths_to_data::Vector{String}, time_period::Tuple{Int64,Int64}, save_directory::String)","category":"page"},{"location":"analysisTools/#ISMwithLSTM.PlotProximity-Tuple{Any, ProxFct, Vector{String}, Tuple{Int64, Int64}, String}","page":"Analysis Tools","title":"ISMwithLSTM.PlotProximity","text":"function PlotProximity(LSTM, pr::ProxFct ,paths_to_data::Vector{String}, time_period::Tuple{Int64,Int64},save_directory::String)\n\nPlots the real Proximity Function vs. the learned one on the given time_period\n\nArguments\n\nLSTM-\npr::ProxFct\npaths_to_data::Vector{String}\ntime_period::Tuple{Int64,Int64}\nsave_directory::String\n\n\n\n\n\n","category":"method"},{"location":"analysisTools/","page":"Analysis Tools","title":"Analysis Tools","text":" PlotOnsetComparsion(LSTM, \n        paths_to_data::Vector{String}, \n        pr::ProxFct,\n        time_period::Tuple{Int64,Int64};\n        t_1= 60::Int, \n        t_2=70::Int,\n        save_directory::String,\n        ylim = (120,190))","category":"page"},{"location":"analysisTools/#ISMwithLSTM.PlotOnsetComparsion-Tuple{Any, Vector{String}, ProxFct, Tuple{Int64, Int64}}","page":"Analysis Tools","title":"ISMwithLSTM.PlotOnsetComparsion","text":"function PlotOnsetComparsion(LSTM, \n    paths_to_data::Vector{String}, \n    pr::ProxFct,\n    time_period::Tuple{Int64,Int64};\n    t_1= 60::Int, \n    t_2=70::Int, \n    save_directory::String)\n\nPlots the predicted onsets vs the actual ones.\n\nArguments\n\nLSTM\npaths_to_data::Vector{String}\npr::ProxFct\ntime_period::Tuple{Int64,Int64}\n\nKeyword Arguments\n\nt_1= 60::Int\nt_2=70::Int\nsave_directory::String\nylim::Tuple{Int64,Int64}\n\n\n\n\n\n","category":"method"},{"location":"analysisTools/","page":"Analysis Tools","title":"Analysis Tools","text":"WriteOutLeadTimeAnalysis(LSTM,  \n    paths_to_data::Vector{String}, \n    pr::ProxFct,\n    save_directory::String; \n    lead_time_range = (60,110),\n    time_period = (1981,2020)::Tuple{Int64,Int64})","category":"page"},{"location":"analysisTools/#ISMwithLSTM.WriteOutLeadTimeAnalysis-Tuple{Any, Vector{String}, ProxFct, String}","page":"Analysis Tools","title":"ISMwithLSTM.WriteOutLeadTimeAnalysis","text":"function WriteOutLeadTimeAnalysis(LSTM, \n    paths_to_data::Vector{String}, \n    pr::ProxFct,\n    save_directory::String; \n    lead_time_range = (60,110),\n    time_period = (1981,2020)::Tuple{Int64,Int64})\n\nWrites out RSME and Correlation of the onset days for the given lead-time range to a .txt file.\n\nArguments\n\nLSTM\npaths_to_data::Vector{String}\npr::ProxFct\nsave_directory::String\n\nKeyword Arguments\n\nlead_time_range = (60,110)\ntime_period = (1981,2020)::Tuple{Int64,Int64})\n\nReturns\n\nvector of standard deviation of the predicted onset days\nvector of correlation between predicted and actual onset days\n\n\n\n\n\n","category":"method"},{"location":"analysisTools/","page":"Analysis Tools","title":"Analysis Tools","text":"PlotLeadTimeAnalysis( \n    path_to_data::String, \n    save_directory::String; \n    lead_time_range = (60,110),\n    time_period = (1981,2020)::Tuple{Int64,Int64})","category":"page"},{"location":"analysisTools/#ISMwithLSTM.PlotLeadTimeAnalysis-Tuple{String, String}","page":"Analysis Tools","title":"ISMwithLSTM.PlotLeadTimeAnalysis","text":"function PlotLeadTimeAnalysis( \n    path_to_data::String, \n    save_directory::String; \n    lead_time_range = (60,110),\n    time_period = (1981,2020)::Tuple{Int64,Int64})\n\nPlots the RSME and Correlation of the onset days for the given lead-time range. Takes in data from the function WriteOutLeadTimeAnalysis\n\nArguments\n\npath_to_data::String\nsave_directory::String\n\nKeyword Arguments\n\nlead_time_range = (60,110)\ntime_period = (1981,2020)::Tuple{Int64,Int64})\n\n\n\n\n\n","category":"method"},{"location":"LSTMTools/","page":"LSTM Tools","title":"LSTM Tools","text":"SetUpLSTM(in_size::Int64,C_size::Int64)","category":"page"},{"location":"LSTMTools/#ISMwithLSTM.SetUpLSTM-Tuple{Int64, Int64}","page":"LSTM Tools","title":"ISMwithLSTM.SetUpLSTM","text":"function SetUpLSTM(in_size::Int64,C_size::Int64)\n\nSets up an LSTM neural network (for reference check out this article) where the input dimension in each recurrent step is given by in_size and the dimension of the cell state by C_size. The dimension of the state which the acticle calls 'h' is the same as the dimension of the cell state, so the final output is computed by a single dense layer mapping 'h' to a scalar value (i.e. the estimate for the proximity function at the given time).\n\n\n\n\n\n","category":"method"},{"location":"LSTMTools/","page":"LSTM Tools","title":"LSTM Tools","text":"trainLSTM(pr::ProxFct,\n    paths_to_data::Vector{String},\n    train_period::Tuple{Int64, Int64}=(1948,1980),\n    test_period::Tuple{Int64, Int64}=(1981,2010);\n    C_dim::Int64 = 5,\n    Tr::Int64 = 365+365,\n    epochs::Int64 = 100,\n    λ1::Float64 = 0.,\n    λ2::Float64 = 0.,\n    learning_rate::Float64 = 1e-2,\n    reduce_learning_rate::Int64 = 20)\n","category":"page"},{"location":"LSTMTools/#ISMwithLSTM.trainLSTM","page":"LSTM Tools","title":"ISMwithLSTM.trainLSTM","text":"function trainLSTM(pr::ProxFct,\n    paths_to_data::Vector{String},\n    train_period::Tuple{Int64, Int64}=(1948,1980),\n    test_period::Tuple{Int64, Int64}=(1981,2010);\n    C_dim::Int64 = 5,\n    Tr::Int64 = 365+365,\n    epochs::Int64 = 100,\n    λ1::Float64 = 0.,\n    λ2::Float64 = 0.,\n    learning_rate::Float64 = 1e-2,\n    reduce_learning_rate::Int64 = 20)\n\nTrains a given LSTM network on training data and hyperparameters specified by the arguments of the function\n\nArguments\n\npr::ProxFct: Instance of ProxFct holding the information about the proximity function\npaths_to_data::Vector{String}: Array of strings describing the paths to the data which should be used for training\ntrain_period::Tuple{Int64, Int64}=(1948,1980): Start and end year of training\ntest_period::Tuple{Int64, Int64}=(1981,2010): Start and end year of testing\nC_dim::Int64 = 5: Dimension of the cell state of the LSTM (for reference check out this article)\nTr::Int64 = 366+365: Transient time in days. The model will not be trained on the first Tr number of days of the training set\nepochs::Int64 = 100: Number of maximum epochs (early stopping possible if loss diverges or loss reaches a platteu)\nλ1::Float64 = 0.: Weight of the L1 regularisation\nλ2::Float64 = 0.: Weight of the L2 regularisation\nlearning_rate::Float64 = 1e-3: Initial learning rate. It reduces by a factor of two every 30 epochs.\nreduce_learning_rate::Int64 = 20: Reduces learing rate by a factor of two in regular steps.\n\nReturns\n\nTrained LSTM\nArray of train losses (one loss value per epoch)\nArray of test losses (one loss value per epoch)\nLoss function: takes arguments (data,prox,LSTM)\n\n\n\n\n\n","category":"function"},{"location":"LSTMTools/","page":"LSTM Tools","title":"LSTM Tools","text":"saveLSTM(LSTM,path::String)","category":"page"},{"location":"LSTMTools/#ISMwithLSTM.saveLSTM-Tuple{Any, String}","page":"LSTM Tools","title":"ISMwithLSTM.saveLSTM","text":"function saveLSTM(LSTM,path::String)\n\nSaves the LSTM as a '.bson' file at the given location.\n\nArguments\n\nLSTM: LSTM one wants to save\npath::String: e.g. \"parameters/my_lstm.bson\"\n\n\n\n\n\n","category":"method"},{"location":"LSTMTools/","page":"LSTM Tools","title":"LSTM Tools","text":"loadLSTM(path::String)","category":"page"},{"location":"LSTMTools/#ISMwithLSTM.loadLSTM-Tuple{String}","page":"LSTM Tools","title":"ISMwithLSTM.loadLSTM","text":"function loadLSTM(path::String)\n\nReturns the LSTM saved at the given location.\n\nArguments\n\npath::String: e.g. \"parameters/my_lstm.bson\"\n\n\n\n\n\n","category":"method"},{"location":"LSTMTools/","page":"LSTM Tools","title":"LSTM Tools","text":"saveFct(fct,path::String)","category":"page"},{"location":"LSTMTools/#ISMwithLSTM.saveFct-Tuple{Any, String}","page":"LSTM Tools","title":"ISMwithLSTM.saveFct","text":"function saveFct(loss,path::String)\n\nSaves the loss function as a '.bson' file at the given location.\n\nArguments\n\nloss: loss function one wants to save\npath::String: e.g. \"losses/my_loss.bson\"\n\n\n\n\n\n","category":"method"},{"location":"LSTMTools/","page":"LSTM Tools","title":"LSTM Tools","text":"loadFct(path::String)","category":"page"},{"location":"LSTMTools/#ISMwithLSTM.loadFct-Tuple{String}","page":"LSTM Tools","title":"ISMwithLSTM.loadFct","text":"function loadFct(path::String)\n\nReturns the Loss function saved at the given location.\n\nArguments\n\npath::String: e.g. \"losses/my_loss.bson\"\n\n\n\n\n\n","category":"method"},{"location":"LSTMTools/","page":"LSTM Tools","title":"LSTM Tools","text":"ProxFct(path_to_data::String)","category":"page"},{"location":"LSTMTools/#ISMwithLSTM.ProxFct-Tuple{String}","page":"LSTM Tools","title":"ISMwithLSTM.ProxFct","text":"function ProxFct(path_to_data::String)\n\nReturns an instance of the struct ProxFct with fields filled with data according to the given path.  This struct stores relevant data for the proximity function. It is overloaded with a function returning  the data of a respective intervall of years.\n\nFields\n\nProxData::Matrix{Real}: Matrix with first column filled with the years and second column filled with the vaules of ΔTT\nonsets::Matrix{Real}: First column with  years and second column with the onset date of the respective year\n\nExamples\n\njulia> pr = ProxFct(\"datafile.txt\")\nProxFct(Real[1948.0 0.0; 1948.0 0.0; … ; 2022.0 0.0; 2022.0 0.0], Real[1948 157; 1949 151; … ; 2021 148; 2022 145])\n\njulia> pr((2010,2020))\n(Real[0.350210970464135, 0.3544303797468354, 0.3586497890295358, 0.36286919831223624, 0.36708860759493667, 0.3713080168776371, \n0.3755274261603375, 0.37974683544303794, 0.38396624472573837, 0.3881856540084388  …  0.3114035087719298, 0.3157894736842105, \n0.3201754385964912, 0.32456140350877194, 0.3289473684210526, 0.3333333333333333, 0.33771929824561403, 0.3421052631578947, \n0.3464912280701754, 0.3508771929824561], Real[155, 159, 152, 147, 161, 158, 161, 150, 151, 160, 156])\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ISMwithLSTM","category":"page"},{"location":"#ISMwithLSTM","page":"Home","title":"ISMwithLSTM","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ISMwithLSTM.","category":"page"},{"location":"","page":"Home","title":"Home","text":"With this package we try to provide the tools it takes to reproduce the results obtained by the research by Mitsui and Boers aiming for a prediction of the onset date of the Indian Summer Monsoon, using an LSTM architecture opposed to the echo state networks used in the reference paper.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
