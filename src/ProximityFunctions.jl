using DelimitedFiles
using Logging

include("HelperFunctions.jl")

struct ProxFct
    ProxData::Matrix{Real}
    onsets::Matrix{Real}
end


"""
    function ProxFct(path_to_data::String)

Returns an instance of the struct ProxFct with fields filled with data according to the given path. 
This struct stores relevant data for the proximity function. It is overloaded with a function returning 
the data of a respective intervall of years.

# Fields

- `ProxData::Matrix{Real}`: Matrix with first column filled with the years and second column filled with the vaules of ΔTT
- `onsets::Matrix{Real}`: First column with  years and second column with the onset date of the respective year

# Examples
```jldoctest
julia> pr = ProxFct("datafile.txt")
ProxFct(Real[1948.0 0.0; 1948.0 0.0; … ; 2022.0 0.0; 2022.0 0.0], Real[1948 157; 1949 151; … ; 2021 148; 2022 145])

julia> pr((2010,2020))
(Real[0.350210970464135, 0.3544303797468354, 0.3586497890295358, 0.36286919831223624, 0.36708860759493667, 0.3713080168776371, 
0.3755274261603375, 0.37974683544303794, 0.38396624472573837, 0.3881856540084388  …  0.3114035087719298, 0.3157894736842105, 
0.3201754385964912, 0.32456140350877194, 0.3289473684210526, 0.3333333333333333, 0.33771929824561403, 0.3421052631578947, 
0.3464912280701754, 0.3508771929824561], Real[155, 159, 152, 147, 161, 158, 161, 150, 151, 160, 156])
```
"""
function ProxFct(path_to_data::String)
    ΔTT_data = readdlm(path_to_data)

    zero_crossing = []
    onset_days = []
    onset_years = []

    for i in unique(ΔTT_data[:,1])
        index_year = findallIndex(x -> x==i,ΔTT_data[:,1])
        ΔTT_yearly = ΔTT_data[index_year,2]
        push!(zero_crossing,FindSignSwitch(ΔTT_yearly).+index_year[1].-1)  
        push!(onset_days, FindSignSwitch(ΔTT_yearly)[1])
        push!(onset_years, Int(i))
    end

    onsets = hcat(onset_years,onset_days)

    zero_crossing = [(zero_crossing...)...] # Flatten AbstractArray

    ProxData = zeros((length(ΔTT_data[:,1]), 2))
    first_index = 157 # This is the first index where ΔTT crosses 0
    for i in 1:(length(ΔTT_data[:,1]))
        ProxData[i,1] = Int64(ΔTT_data[i,1]) # Adding the years
        if i < first_index
            ProxData[i,2] = 0 # Adding dummy values
        else
            ProxData[i,2] = LinearProximityFunction(i, zero_crossing)
       end
    end

    return ProxFct(ProxData,onsets)
end



function (pr::ProxFct)(time_period::Tuple{Real, Real})
    if (time_period[1] < pr.ProxData[1,1]) | (time_period[2] > pr.ProxData[end,1])
        @warn "Given timeperiod is not entirely included in data"
    end
    (time_period[1] > time_period[2]) && error("The years in the tuple are in the wrong order")
    cond(i) = ((i >= time_period[1]) & (i <= time_period[2])) ? true : false
    indices_ProxData = findallIndex(cond,pr.ProxData[:,1])
    indices_onsets = findallIndex(cond,pr.onsets[:,1])
    return pr.ProxData[indices_ProxData,2], pr.onsets[indices_onsets,2]
end

(pr::ProxFct)(year::Real) = pr((year,year))


"""
    function LinearProximityFunction(x::Int, zero_crossing)

Returns value of the linear Proximity Function proposed by [Mitsui, Boers](https://iopscience.iop.org/article/10.1088/1748-9326/ac0acb/meta)

# Arguments
- `x::Int`
- `path_to_zero_crossing::String`: path to the zero_crossings.txt file

# Returns
- `::Float64`

"""
function LinearProximityFunction(x::Int, zero_crossing)
    for i in 1:(length(zero_crossing)-1)
        x_1 = zero_crossing[i]
        x_2 = zero_crossing[i+1]
        if x in (x_1:1:x_2)
            # Differ between rising and falling flank
            if (i % 2) == 1 
                return  1/(x_1-x_2)*(x-x_1) + 1
            else
                return 1/(x_2-x_1)*(x-x_1)
            end 
        end
    end
    return 0. # Dummy value because this correpsonds to the data after the last zero crossing. This is not used for training anyways.
end


