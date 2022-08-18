using DelimitedFiles

"""
function LinearProximityFunction(x::Int)
Returns value of the linear Proximity Function proposed by [Mitsui, Boers](https://iopscience.iop.org/article/10.1088/1748-9326/ac0acb/meta)

# Arguments
* `x::Int`
* `path_to_zero_crossing::String, path to the zero_crossings.txt file``

# Returns
* `::Float64`

"""
function LinearProximityFunction(x::Int, path_to_zero_crossing::String)
    file_path = path_to_zero_crossing 
    index_sign = []
    try
        index_sign = Int.(readdlm(file_path))
    catch ArgumentError
        error("File does not exist in the given path. You have to first create it using WriteSignSwitchesIndices \n Abort.")
    end
    for i in 1:(length(index_sign)-1)
        x_1 = index_sign[i]
        x_2 = index_sign[i+1]
        if x in (x_1:1:x_2)
            # Differ between rising and falling flank
            if (i % 2) == 1 
                return  1/(x_1-x_2)*(x-x_1) + 1
            else
                return 1/(x_2-x_1)*(x-x_1)
            end 
        end
    end
    return 0 # Dummy value because this correpsonds to the data after the last zero crossing. This is not used for training anyways.
end


