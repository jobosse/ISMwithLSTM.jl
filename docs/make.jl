using ISMwithLSTM
using Documenter

DocMeta.setdocmeta!(ISMwithLSTM, :DocTestSetup, :(using ISMwithLSTM); recursive=true)

makedocs(;
    modules=[ISMwithLSTM],
    authors="Johannes Bosse <johannes.bosse@rwth-aachen.de>",
    repo="https://github.com/jobosse/ISMwithLSTM.jl/blob/{commit}{path}#{line}",
    sitename="ISMwithLSTM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Analysis Tools" => "analysisTools.md",
        "LSTM Tools" => "LSTMTools.md"
    ],


)

deploydocs(
    repo = "git@github.com:jobosse/ISMwithLSTM.jl.git",
)
