"""
findallIndex(condition, x::AbstractArray)

Returns the indices of the values that fulfill the condition.

# Returns
* results::Vector{Int64} which contains the indices.
"""
function findallIndex(condition, x::AbstractVector{T} where T<:Real)
       results = Int[]

       for i in eachindex(x)
           if condition(x[i])
               push!(results, i)
           end
       end
       return results
end

"""
    function FindSignSwitch(x::AbstractVector{T} where T<:Real)
Finds the zero-crossing points in the given data. Used for the ΔTT data.

#Arguments
* `x::AbstractVector{T} where T<:Real`

#Returns
* `vector containing indices of the zero-crossing-points`
"""    
function FindSignSwitch(x::AbstractVector{T} where T<:Real)
    x_sign = @. sign(x)
    result = Int[]
    
    for i in (1:length(x)-1)
        if x_sign[i]*x_sign[i+1] < 0
            push!(result,i+1)
        end
    end
    result = [result[1],result[end]] # only save the first and last index where the zero is crossed.
    return result
end