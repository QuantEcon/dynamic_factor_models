using ExcelReaders
using Statistics
import Dates
using StatsBase
using LinearAlgebra

abstract type DataFrequency end
struct MonthlyData <: DataFrequency
    nobs::Int # number of observations
    ns::Int # number of series
end

struct QuarterlyData <: DataFrequency
    nobs::Int # number of observations
    ns::Int # number of series
end

abstract type DetrendMethod end
struct BiWeight{TR <: Real} <: DetrendMethod
    weight::TR
end
struct Mean <: DetrendMethod end
struct NoDetrend <: DetrendMethod end

abstract type DataType end
struct RealData <: DataType end
struct AllData <: DataType end
get_sample_periods(initvec::Array, lastvec::Array, ::MonthlyData) =
    12*(lastvec[1]-initvec[1]-1)+lastvec[2]+(12-initvec[2]+1)
get_sample_periods(initvec::Array, lastvec::Array, ::QuarterlyData) =
    4*(lastvec[1]-initvec[1]-1)+lastvec[2]+(4-initvec[2]+1)
function MonthlyData(initvec, lastvec, ns)
    nobs = 12*(lastvec[1]-initvec[1]-1)+lastvec[2]+(12-initvec[2]+1)
    return MonthlyData(nobs, ns)
end
function QuarterlyData(initvec, lastvec, ns)
    nobs = 4*(lastvec[1]-initvec[1]-1)+lastvec[2]+(4-initvec[2]+1)
    return QuarterlyData(nobs, ns)
end
"""
deflate series by PCE
"""
function deflate_series!(labvec_long::AbstractVector, labvec_short::AbstractVector,
                        x::AbstractVector, price_def::AbstractVector,
                        price_def_lfe::AbstractVector, price_def_pgdp,
                        is::Integer, defcode::Val{1})
    x .= x./price_def
    labvec_long[is] = labvec_long[is] * " Defl by PCE Def"
    labvec_short[is] = "Real_" * labvec_short[is]
    return x
end
"""
deflate series by PCE(LFE)
"""
function deflate_series!(labvec_long::AbstractVector, labvec_short::AbstractVector,
                         x::AbstractVector, price_def::AbstractVector,
                         price_def_lfe::AbstractVector, price_def_pgdp,
                         is::Integer, defcode::Val{2})
    x .= x./price_def_lfe
    labvec_long[is] = labvec_long[is] * " Defl by PCE(LFE) Def"
    labvec_short[is] = "Real_" * labvec_short[is]
    return x
end
"""
deflate series by GDP
"""
function deflate_series!(labvec_long::AbstractVector, labvec_short::AbstractVector,
                         x::AbstractVector, price_def::AbstractVector,
                         price_def_lfe::AbstractVector, price_def_pgdp,
                         is::Integer, defcode::Val{3})
    x .= x./price_def_pgdp
    labvec_long[is] = labvec_long[is] * " Defl by GDP Def"
    labvec_short[is] = "Real_" * labvec_short[is]
    return x
end
deflate_series!(labvec_long::AbstractVector, labvec_short::AbstractVector,
                x::AbstractVector, price_def::AbstractVector,
                price_def_lfe::AbstractVector, price_def_pgdp,
                is::Integer, defcode::Val{0}) = x
"""
transform monthly data to quarterly data by averaging
##### Arguments
- `data_m::AbstractArray`: monthly data. `size(y, 1) == length(date)`
- `date_m::AbstractVector{Date}`: Vector of Date.
"""
function monthly_to_quarterly(data_m::AbstractArray, date_m::AbstractVector{Dates.Date}, ::MonthlyData)
    if size(data_m, 1) != length(date_m)
        @error "number of rows of data_m must be same as the length of date"
    end
    years = Dates.year.(date_m)
    date_m_quarter = transform_date.(date_m)
    T_q = length(unique(date_m_quarter))
    data_q = Array{Union{Missing, Float64}}(undef, T_q, size(data_m, 2))
    for t = 1:T_q
        id = (date_m_quarter .== [unique(date_m_quarter)[t]])
        data_q[t, :] = mean(data_m[id, :], dims=1)
    end
    return data_q, unique(date_m_quarter)
end
function monthly_to_quarterly(data_q::AbstractArray, date_q::AbstractVector{Dates.Date}, ::QuarterlyData)
    date_q_quarter = transform_date.(date_q)
    return data_q, date_q_quarter
end
transform_date(date::Dates.Date) = (Dates.year(date), get_quarter(date))
get_quarter(date::Dates.Date) = cld(Dates.month(date), 3)

# Level
transform(x::AbstractVector, ::Val{1}) = x
# First Difference
transform(x::AbstractVector, ::Val{2}) = vcat(missing, x[2:end].-x[1:end-1])
# Second Difference
transform(x::AbstractVector, ::Val{3}) = vcat(missing, missing, x[3:end]-2*x[2:end-1].+x[1:end-2])
# Log-level
transform(x::AbstractVector, ::Val{4}) = log.(x)
# Log-First-Difference
transform(x::AbstractVector, ::Val{5}) = transform(log.(x), Val(2))
# Log-Second-Difference
transform(x::AbstractVector, ::Val{6}) = transform(log.(x), Val(3))

function transform!(data::AbstractMatrix, tcode::AbstractVector)
    if size(data, 2) != length(tcode)
        @error "size(data, 2) must equal to length(tcode)"
    end
    for i = 1:size(data, 2)
        data[:, i] = transform(data[:, i], Val(tcode[i]))
    end
    return nothing
end
# Threshold multiple for IQR is 4.5
adjust_outlier!(x::AbstractVector, ::Val{1}, io_method::Integer) =
    adjust_outlier!(x, 4.5, io_method)
# Threshold multiple for IQR is 3
adjust_outlier!(x::AbstractVector, ::Val{2}, io_method::Integer) =
    adjust_outlier!(x, 3, io_method)
adjust_outlier!(x::AbstractVector, ::Val{0}, io_method::Integer) = nothing

function adjust_outlier!(x::AbstractVector, thr::Real, tflag::Integer)
    # compute median and IQR
    zm = median(skipmissing(x))
    iqr = quantile(skipmissing(x), 0.75) - quantile(skipmissing(x), 0.25)
    z = x[.!ismissing.(x)]
    # sort!(z)
    # tmp = z[ceil.(Int, length(z)*[1/4, 1/2, 3/4])]
    # zm = tmp[2]
    # iqr = tmp[3]-tmp[1]
    (iqr >= 1e-6) || @error "error in adjusting outlier"

    ya = abs.(x .- zm)
    i_outlier = ya .> (thr*iqr)
    adjust_x!(x, i_outlier, zm, thr, iqr, Val(tflag))
    return nothing
end
"""
replace with missing value
"""
function adjust_x!(x::AbstractVector, i_outlier, zm, thr, iqr, ::Val{0})
     x[i_outlier] = missing
     return nothing
end
"""
replace with maximum or minimum value
"""
function adjust_x!(x, i_outlier, zm, thr, iqr, ::Val{1})
    isign = Int.(x[i_outlier] .> 0) - Int.(x[i_outlier] .< 0)
    yt = zm .+ isign.*(thr*iqr)
    x[i_outlier] = yt
    return nothing
end
"""
replace with median value
"""
function adjust_x!(x, i_outlier, zm, thr, iqr, ::Val{2})
    x[i_outlier] = zm
    return nothing
end
"""
replace with local median (obs + or - 3 on each side)
"""
function adjust_x!(x, i_outlier, zm, thr, iqr, ::Val{3})
    # Compute rolling median
    iwin = 3  # Window on either side
    for i in findall(i_outlier)
        j1 = max(1, i-iwin)
        j2 = min(length(x), i+iwin)
        x[i] = median(skipmissing(x[j1:j2]))
    end
    return nothing
end
"""
replace with one-sided median (5 preceding obs)
"""
function adjust_x!(x, i_outlier, zm, thr, iqr, ::Val{4})
    # Compute rolling median
    iwin=5  # Window on one side
    for i in findall(i_outlier .& .!ismissing.(i_outlier))
        j1 = max(1, i-iwin)
        j2 = i
        x[i] = median(skipmissing(x[j1:j2]))
    end
    return nothing
end

readin_data(frequency::MonthlyData) =
    readin_monthly_data(true, 4, 2, 6, [1, 2, 3, 5], frequency)
readin_data(frequency::QuarterlyData) =
    readin_monthly_data(true, 4, 2, 5, [1, 2, 3, 5], frequency)
readdata(::MonthlyData) = readxlsheet("data/hom_fac_1.xlsx", "Monthly")
readdata(::QuarterlyData) = readxlsheet("data/hom_fac_1.xlsx", "Quarterly")
function readin_monthly_data(correct_outlier::Bool,
                             io_method::Integer, # Replacement of outliers,
                             ndesc::Integer,     # number of "description" rows in Excel file
                             ncodes::Integer,    # number of rows of "codes" in Excel file
                             cat_include::AbstractVector,
                             frequency::DataFrequency,
                             )
    dnobs = frequency.nobs
    ns_m = frequency.ns
    ## read monthly data
    wholedata = readdata(frequency)

    maindata = wholedata[1:1+ndesc+ncodes+dnobs, 2:ns_m+1]
    date = Dates.Date.(wholedata[1+ndesc+ncodes+1:1+ndesc+ncodes+dnobs, 1])

    namevec, labvec_long, labvec_short, aggcode, tcode, defcode,
           outliercode, includecode, catcode = get_headers(maindata, frequency)
    datamat_m = maindata[1+ndesc+ncodes+1:end, :]
    datamat_m[.!isa.(datamat_m, Float64)] .= missing # set missing values
    datamat_m = convert(Array{Union{Missing, Float64}, 2}, datamat_m)

    # Price Deflators
    price_def, price_def_lfe, price_def_pgdp =
        get_deflators(namevec, datamat_m, frequency)

    # Standardize Killian Activity Index
    standardize_killian!(datamat_m, namevec, frequency)

    usedcols_id = (includecode .!=0) .& [in(cat, cat_include) for cat in floor.(catcode)]
    data_m = datamat_m[:, usedcols_id]
    bpdefcode = defcode[usedcols_id]
    bpoutlier = outliercode[usedcols_id]
    bplab_long = labvec_long[usedcols_id]
    bplab_short = labvec_short[usedcols_id]
    ns = size(data_m, 2)

    [deflate_series!(bplab_long, bplab_short, view(data_m, :, i),
                     price_def, price_def_lfe, price_def_pgdp, i, Val(bpdefcode[i])) for i =1:ns]
    data_q, date_q = monthly_to_quarterly(data_m, date, frequency)  # Temporally aggregated to quarterly
    bpdata_raw = copy(data_q)
    transform!(data_q, tcode[usedcols_id]) # Transform .. log, first difference, etc.
    bpdata_noa = copy(data_q)
    !correct_outlier || [adjust_outlier!(view(data_q, :, i), Val(bpoutlier[i]), io_method) for i = 1:ns]
    return data_q, bpdata_raw, bpdata_noa, date_q, catcode[usedcols_id],
           includecode[usedcols_id], namevec[usedcols_id]
end

function get_headers(maindata, ::MonthlyData)
    namevec = uppercase.(convert(Vector{String}, maindata[1, :]))
    labvec_long = convert(Vector{String}, maindata[2, :])   # Vector of "long" labels
    labvec_short = convert(Vector{String}, maindata[3, :])  # Vector of "short" labels
    aggcode = Int.(maindata[4, :]) # Temporal aggregation code
    tcode = Int.(maindata[5, :])   # transformation code
    defcode = Int.(maindata[6, :]) # code for price deflation (nominal to real)
    outliercode = Int.(maindata[7, :]) # code for outlier adjustment
    includecode = Int.(maindata[8, :]) # code for use in factor estimation
    catcode = maindata[9, :]     # category code for ordering variables
    return namevec, labvec_long, labvec_short, aggcode, tcode, defcode,
           outliercode, includecode, catcode
end

function get_headers(maindata, ::QuarterlyData)
    namevec = uppercase.(convert(Vector{String}, maindata[1, :]))
    labvec_long = convert(Vector{String}, maindata[2, :])   # Vector of "long" labels
    labvec_short = convert(Vector{String}, maindata[3, :])  # Vector of "short" labels
    tcode = Int.(maindata[4, :])   # transformation code
    defcode = Int.(maindata[5, :]) # code for price deflation (nominal to real)
    outliercode = Int.(maindata[6, :]) # code for outlier adjustment
    includecode = Int.(maindata[7, :]) # code for use in factor estimation
    catcode = maindata[8, :]     # category code for ordering variables
    return namevec, labvec_long, labvec_short, nothing, tcode, defcode,
           outliercode, includecode, catcode
end

function get_deflators(namevec, datamat, ::MonthlyData)
    j = findall(namevec.=="PCEPI")[1] # PCE Price Deflator
    price_def = datamat[:, j]
    j = findall(namevec.=="PCEPILFE")[1] # PCE-xFE Price Deflator
    price_def_lfe = datamat[:, j]
    return price_def, price_def_lfe, NaN
end

function get_deflators(namevec, datamat, ::QuarterlyData)
    j = findall(namevec.=="PCECTPI")[1] # PCE Price Deflator
    price_def = datamat[:, j]
    j = findall(namevec.=="JCXFE")[1] # PCE Excl. food and energy
    price_def_lfe = datamat[:, j]
    j = findall(namevec.=="GDPCTPI")[1] # GDP Deflator
    price_def_pgdp = datamat[:, j]
    return price_def, price_def_lfe, price_def_pgdp
end

"""
Standardize Killian Activity Index
"""
function standardize_killian!(datamat_m, namevec, ::MonthlyData)
    j = findall(namevec.=="GLOBAL_ACT")[1] # Killian Index
    tmp = datamat_m[:, j]
    tmp1 = tmp[.!ismissing.(tmp)]
    tmp2 = (tmp1.-mean(tmp1))./std(tmp1)
    datamat_m[.!ismissing.(tmp), j] = tmp2
    return nothing
end

standardize_killian!(datamat_m, namevec, ::QuarterlyData) = nothing

function detrend_var!(data::AbstractArray, bw::BiWeight)
    trend = Array{Union{Missing, Float64}}(undef, size(data))
    for i = 1:size(data, 2)
        trend[:, i] = bi_weight_filter(data[:, i], bw.weight)
        data[:, i] = data[:, i] - trend[:, i]
    end
    return trend
end


function detrend_var!(data::AbstractArray, ::Mean)
    trend = repmat(mean(skipmissing(data), dims=1), size(data, 1), 1)
    data .= data .- trend
    return trend
end

detrend_var!(data::AbstractArray, ::NoDetrend) = fill(missing, size(data))

function bi_weight_filter(y::AbstractVector, weight::Real)
    T = length(y)
    ytrend = Array{Union{Missing, Float64}}(undef, T)
    trend = 1:T
    for t = findall(.!ismissing.(y))
        dt = (trend .- t)/weight
        bw_weight = 15/16*((1 .-dt.^2).^2) # bi-weight
        bw_weight[abs.(dt).>=1] .= 0
        bw_weight_ignore_missing = bw_weight[.!ismissing.(y)]
        bw_weight_ignore_missing = bw_weight_ignore_missing ./ sum(bw_weight_ignore_missing)
        ytrend[t] = dot(bw_weight_ignore_missing, y[.!ismissing.(y)])
    end
    return ytrend
end

"""
##### Arguments
- `detrend_method::DetrendMethod`: method for detrending.
- `::RealData`: type of read data. (TODO: `::AllData`)
"""
function readin_data(md::MonthlyData, qd::QuarterlyData,
                     detrend_method::DetrendMethod, ::RealData)
    data_m, bpdata_raw_m, bpdata_noa_m, date_m, bpcatcode_m, inclcode_m, namevec_m =
        readin_data(md)

    data_q, bpdata_raw_q, bpdata_noa_q, date_q, bpcatcode_q, inclcode_q, namevec_q =
        readin_data(qd)

    all(date_m .== date_q) ||
        @error "inconsistent sample size for monthly and quarterly data"

    id = sortperm(vcat(bpcatcode_m, bpcatcode_q))
    bpdata = hcat(data_m, data_q)[:, id]
    bpdata_unfiltered = copy(bpdata)
    bpdata_trend = detrend_var!(bpdata, detrend_method)

    dataset = (bpdata_raw = hcat(bpdata_raw_m, bpdata_raw_q)[:, id],
               bpcatcode =vcat(bpcatcode_m, bpcatcode_q)[id],
               bpdata = bpdata,
               bpdata_unfiltered =bpdata_unfiltered,
               bpdata_noa = hcat(bpdata_noa_m, bpdata_noa_q)[:, id],
               bpdata_trend = bpdata_trend,
               inclcode = vcat(inclcode_m, inclcode_q)[id],
               bpnamevec = vcat(namevec_m, namevec_q)[id],
               calvec = calender_value.(date_q),
               calds = date_q)
    return dataset
end

calender_value(year::Integer, quarter::Integer) = year+(quarter-1)/4
calender_value(calvec::Tuple{Int, Int}) = calender_value(calvec[1], calvec[2])
