module ISMwithLSTM

export LinearProximityFunction
export LoadAnnualData
export SetUpLSTM
export regroupData
export periodicForcing
export RunLSTM
export saveLSTM
export loadLSTM
export saveFct
export loadFct
export trainLSTM
export PlotProximity
export OnsetDayPrediction
export CalculateLoss
export PlotOnsetComparsion
export ProxFct
export WriteOutLeadTimeAnalysis
export PlotLeadTimeAnalysis
export PlotTrainYrsAnalysis


include("ProximityFunctions.jl")
include("ReadInputData.jl")
include("AnalysisTools.jl")
include("LSTM.jl")

end
