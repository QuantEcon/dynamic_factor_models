
# using DataFrames
using Statistics
using DelimitedFiles
using LinearAlgebra
abstract type EstimationMethod end
struct NonParametric <: EstimationMethod end
struct Parametric <: EstimationMethod end

"""
VAR model can be expressed in state-space form:
```math
y_t = Qz_t
```
```
z_t = Mz_{t-1}+Gu_t
```
where ``z_t`` has ``y_t`` and lagged ``y_t``s
"""
struct VARModel{TA <: AbstractArray, TAm <: AbstractArray, TMm <: AbstractMatrix}
    y::TA
    nlag::Int
    withconst::Bool
    initperiod::Int
    lastperiod::Int
    T::Int
    ns::Int
    resid::TAm
    betahat::TAm
    M::TMm
    Q::TMm
    G::TMm
    seps::TMm
end
mutable struct FactorEstimateStats
    T::Int # number of data periods used for factor estimation
    ns::Int # number of data series
    nobs # total number of observations (=sum_i T_i)
    tss
    ssr
    R2::Vector{Union{Missing, Float64}}
end
struct DFMModel
    data::AbstractArray
    inclcode::Vector{Int}
    T::Int  # number of whole data periods
    ns::Int
    nt_min_factor_estimation::Int
    nt_min_factorloading_estimation::Int
    initperiod::Int
    lastperiod::Int
    nfac_o::Int
    nfac_u::Int
    nfac_t::Int
    tol::Float64
    fes::FactorEstimateStats
    factor::AbstractArray
    lambda::AbstractArray
    uar_coef::AbstractArray
    uar_ser::Vector{Float64}
    n_uarlag::Int
    n_factorlag::Int
    factor_var_model::VARModel
end
"""
Note: `factor` and `facor_var_model.y` are actually using same memory.
"""
function DFMModel(data, inclcode,
    nt_min_factor_estimation::Integer, nt_min_factorloading_estimation::Integer,
    initperiod::Integer, lastperiod::Integer,
    nfac_o::Integer, nfac_u::Integer, tol, n_uarlag::Integer, n_factorlag::Integer)
    size(data, 2) == length(inclcode) || error("length of inclcode must equal to number of data series")
    initperiod < lastperiod || error("initperiod must be smaller than lastperiod")
    ((n_uarlag > 0) && (n_factorlag > 0)) || error("n_uarlag and n_factorlag must be positive")

    T, ns = size(data)
    nfac_t = nfac_o+nfac_u
    # if ns == 207
    #     fes = FactorEstimateStats(lastperiod - initperiod + 1,
    #                           length(inclcode.==1),
    #                           missing, missing, missing,
    #                           Vector{Union{Missing, Float64}}(undef, length(inclcode)))
    # else
        fes = FactorEstimateStats(lastperiod - initperiod + 1,
                              count(inclcode.==1),
                              missing, missing, missing,
                              Vector{Union{Missing, Float64}}(undef, count(inclcode.==1)))
    # end
    factor = Matrix{Union{Missing, Float64}}(missing, T, nfac_t)
    lambda = Matrix{Float64}(undef, ns, nfac_t)
    uar_coef = Matrix{Float64}(undef, ns, n_uarlag)
    uar_ser = Vector{Float64}(undef, ns)
    factor_var_model = VARModel(factor, n_factorlag, initperiod= initperiod,
                                lastperiod=lastperiod)
    return DFMModel(data, vec(inclcode), T, ns,
                    nt_min_factor_estimation, nt_min_factorloading_estimation,
                    initperiod, lastperiod, nfac_o, nfac_u, nfac_t,
                    tol, fes, factor, lambda, uar_coef, uar_ser,
                    n_uarlag, n_factorlag, factor_var_model)
end

function drop_missing_row(A::AbstractMatrix)
    tmp = .!any(ismissing.(A), dims=2)
    return A[vec(tmp), :], tmp
end

function drop_missing_col(A::AbstractMatrix)
    tmp = drop_missing_row(A')
    return tmp[1]', tmp[2]'
end

function pca_score(X, nfac_u::Integer)
    _, _, V = svd(X)
    score = (X*V)[:, 1:nfac_u]
    return score
end

"""
##### Arguments
- `y::AbstractVector`: length `T` Vector or `TxN` Matrix where `T`
                       is sample size and `N` is the number of
                       regressed variables
- `X::AbstractArray`: `TxK` Matrix where `K` is the number of
                      regressors
##### Outputs
- `b`: OLS estimator of the coefficients
- `e`: residual
"""
function ols(y::AbstractArray, X::AbstractArray)
    b = X\y
    e = y - X*b
    return b, e
end

abstract type OLSSkipRule end
struct Balanced <: OLSSkipRule end
struct Unbalanced <: OLSSkipRule end
"""
do OLS ignoring samples containing missing values
##### Outputs
- `b`: OLS estimator of the coefficients
- `e`: residual
- `numrow`: BitArray inidicating rows used to estimate
"""
function ols_skipmissing(y::AbstractMatrix, X::AbstractArray, ::Balanced)
    N = size(y, 2)
    tmp, numrow = drop_missing_row([y X])
    y_used, x_used = tmp[:, 1:N], tmp[:, N+1:end]
    b, e = ols(y_used, x_used)
    return b, e, vec(numrow)
end
function ols_skipmissing(y::AbstractVector, X::AbstractArray, method::Balanced)
    b, e, numrow = ols_skipmissing(reshape(y, size(y, 1), size(y, 2)), X, method)
    return b, vec(e), numrow
end
"""
##### Arguments
- `y::AbstractMatrix`: `TxN`
- `X::AbstractArray`: `TxK` Matrix or `T` Vector
"""
function ols_skipmissing(y::AbstractMatrix, X::AbstractArray, ::Unbalanced)
    if size(y, 1) != size(X, 1)
        error("Sample size must be same")
    end
    T, N = size(y)
    b = Matrix{Float64}(undef, size(X, 2), N)
    e = Matrix{Union{Missing, Float64}}(missing, T, N)
    numrow = BitArray(undef, T, N)
    for i=1:N
        tmp = ols_skipmissing(y[:, i], X, Balanced())
        b[:, i] = tmp[1]
        e[tmp[3], i] = tmp[2]
        numrow[:, i] = tmp[3]
    end
    return b, e, numrow
end

function lagmat(X::AbstractArray, lags::AbstractVector)
    nc = size(X, 2)
    Xlag = Matrix{Union{Missing, Float64}}(missing, size(X, 1), nc*length(lags))
    for (i, lag) in enumerate(lags)
        Xlag[lag+1:end, nc*(i-1)+1:nc*i] .= X[1:end-lag, :]
    end
    return Xlag
end
lagmat(X::AbstractArray, lag::Integer) = lagmat(X, [lag])

function uar(y::AbstractVector, n_lags::Integer)
    x = lagmat(y, 1:n_lags)
    arcoef, ehat, _ = ols_skipmissing(y, x, Balanced())
    ssr = dot(ehat, ehat)
    ser = sqrt(ssr/(size(x, 1)-size(x, 2)))
    return arcoef, ser
end
"""
estimate factor by iteration using balanced data
"""
function estimate_factor!(m::DFMModel, max_iter::Integer=100000000,
                         computeR2::Bool=true)
    data = m.data
    initperiod, lastperiod, nt_min, nfac_u, nfac_o, tol =
        m.initperiod, m.lastperiod, m.nt_min_factor_estimation,
        m.nfac_u, m.nfac_o, m.tol
    # use part of the data
    est_data = data[:, m.inclcode.==1]
    xdata = est_data[initperiod:lastperiod, :]

    # preprocess data to have unit standard error
    xdata_standardized, xdatastd = standardize_data(xdata)

    m.fes.tss = sum(skipmissing(xdata_standardized.^2))
    m.fes.nobs = length(xdata_standardized[.!ismissing.(xdata_standardized)])

    xbal, _ = drop_missing_col(xdata_standardized)

    # Get initial F_t given Lambda_t using PCA
    f = pca_score(xbal, nfac_u)
    m.fes.ssr = 0
    diff = 1000
    lambda = Array{Union{Missing, Float64}}(undef, m.fes.ns, m.nfac_t)
    for iter = 1:max_iter
        ssr_old = m.fes.ssr
        # given F_t, get Lambda_t
        for i = 1:m.fes.ns
            tmp, usedrows = drop_missing_row([xdata_standardized[:, i] f])
            if count(usedrows) >= nt_min
                lambda[i, :] =
                    ols_skipmissing(xdata_standardized[:, i], f, Balanced())[1]'
            end
        end
        # given Lambda_t, get F_t by regressing X_t on Lambda_t for each t
        tmp = ols_skipmissing(xdata_standardized', lambda[:, nfac_o+1:end], Unbalanced())
        f, ehat = tmp[1]', tmp[2]
        m.fes.ssr = sum(sum(skipmissing(ehat.^2)))
        diff = abs(ssr_old - m.fes.ssr)
        diff >= tol*m.fes.T*m.fes.ns || break
        # println("diff = ", diff)
    end
    m.factor[initperiod:lastperiod,  :] = f
    if computeR2
        for i=1:m.fes.ns
            tmp = drop_missing_row([xdata_standardized[:, i] f])[1]
            if size(tmp, 1) >= nt_min
                _, ehat = ols(tmp[:, 1], tmp[:, 2:end])
                m.fes.R2[i] = compute_r2(tmp[:, 1], ehat)[1]
            end
        end
    end
    return nothing
end

function estimate_factor_loading!(m::DFMModel)
    data, initperiod, lastperiod, fac, nt_min, nfac_t, n_uarlag =
        m.data, m.initperiod, m.lastperiod, m.factor,
        m.nt_min_factorloading_estimation, m.nfac_t, m.n_uarlag
    n_series = size(data, 2)
    r2_mat = Vector{Float64}(undef, n_series)
    for is = 1:n_series
        tmp, numrow = drop_missing_row([data[initperiod:lastperiod, is] fac[initperiod:lastperiod, :]])
        if count(numrow) >= nt_min # if available sample size is large enough
            b, uhat = ols(tmp[:, 1], [tmp[:, 2:end] ones(count(numrow))])
            y_used = data[initperiod:lastperiod, is][vec(numrow), :]
            m.lambda[is, :] .= b[1:end-1]
            r2_mat[is], _, _ = compute_r2(y_used, uhat)
            if r2_mat[is] < 0.9999
                arcoef, ser = uar(uhat, n_uarlag)
            else
                arcoef, ser = zeros(n_uarlag, 1), 0.0
            end
        end
        m.uar_coef[is, :] = arcoef'
        m.uar_ser[is] = ser
    end
    return nothing
end
function VARModel(y::AbstractArray, nlag::Integer=1;
                  withconst::Bool=true,
                  initperiod::Integer=1, lastperiod::Integer=size(y, 1))
    T, ns = size(y, 1), size(y, 2)
    resid = Array{Union{Missing, Float64}}(missing, size(y))
    betahat = Matrix{Union{Missing, Float64}}(missing, ns*nlag+withconst, ns)
    M = Matrix{Union{Missing, Float64}}(missing, ns*nlag, ns*nlag)
    Q = Matrix{Union{Missing, Float64}}(missing, ns, ns*nlag)
    G = Matrix{Union{Missing, Float64}}(missing, ns*nlag, ns)
    seps = Matrix{Union{Missing, Float64}}(missing, ns, ns)
    return VARModel(y, nlag, withconst, initperiod, lastperiod, T, ns, resid, betahat, M, Q, G, seps)
end
function estimate_var!(varm::VARModel, compute_matrices::Bool=true)
    initperiod, lastperiod = varm.initperiod, varm.lastperiod
    withconst, nlag = varm.withconst, varm.nlag
    resid, seps = varm.resid, varm.seps

    y_restricted = varm.y[initperiod:lastperiod, :]

    # regressors
    withconst || (x = lagmat(y_restricted, 1:nlag))
    !withconst || (x = [ones(lastperiod-initperiod+1) lagmat(y_restricted, 1:nlag)])

    # do OLS ignoring the samples containing NaN
    betahat, ehat, numrows = ols_skipmissing(y_restricted, x, Balanced())
    varm.betahat .= betahat

    T_used = count(numrows) # used sample size
    K = size(x, 2) # number of regressors

    ndf = T_used - K # degree of freedom (T-K)
    seps .= ehat'*ehat/ndf # covariance matrix of error term
    resid[initperiod- 1 .+ findall(numrows), :] .= ehat

    !compute_matrices || fill_matrices!(varm, betahat)
    return nothing
end
function fill_matrices!(varm::VARModel, betahat::Array)
    ns, nlag = varm.ns, varm.nlag
    M, Q, G = varm.M, varm.Q, varm.G

    b = betahat[2:end, :]' # now, each row corresponds to each equation

    M .= zeros(ns*nlag, ns*nlag)
    M[1:ns, :] .= b # coefficients of VAR
    M[ns+1:end, 1:end-ns] .= Matrix{Float64}(I, ns*nlag-ns, ns*nlag-ns)　# lag part

    Q .= zeros(ns, ns*nlag)
    Q[1:ns, 1:ns] .= Matrix{Float64}(I, ns, ns)
    G .= zeros(ns*nlag, ns)
    G[1:ns, 1:ns] .= ((cholesky(varm.seps)).U)'
    return nothing
end

function standardize_data(data::AbstractArray)
    datamean = [mean(collect(skipmissing(data[:, i]))) for i = 1:size(data, 2)]'
    # # make correction (which I don't understand why being needed)
    tmp = size(data, 1) .- sum(ismissing.(data), dims = 1)
    tmp = (tmp.-1)./tmp
    datastd = [std(collect(skipmissing(data[:, i]))) for i = 1:size(data, 2)]'.*(tmp.^.5)
    data_standardized = (data .- datamean)./datastd
    return data_standardized, datastd
end

"""
estimate DFM Model non-parametrically
"""
function estimate!(m::DFMModel, ::NonParametric=NonParametric())

    # estimate factor using balanced data
    estimate_factor!(m)

    # estimate factor loading using full sample
    estimate_factor_loading!(m)

    # estimate the equation of factor evolution
    estimate_var!(m.factor_var_model)

    return nothing
end
compute_series(dfmm::DFMModel, is::Integer) = dfmm.factor*dfmm.lambda[is, :]
detrended_year_growth(y::AbstractVector) = vec(sum(lagmat(y, 0:3), dims=2))

find_row_number(date::Tuple{Int, Int}, dates) =
    findall([date == dataset.calds[i] for i=1:length(dataset.calds)])[1]

function compute_r2(y::AbstractArray, e::AbstractVector)
    ssr = dot(e, e)
    tss = dot(y.-mean(y), y.-mean(y))
    return 1-(ssr/tss), ssr, tss
end
function compute_bw_weight(bw_para::Integer)
    bw_weight = zeros(2bw_para+1)
    trend = Vector{Int64}(undef, 2bw_para+1)
    for i=0:100
        trend[bw_para+1+i] = i
        trend[bw_para+1-i] = -i
        bw_weight[bw_para+1+i] = 15/16*(1-(trend[bw_para+1+i]/bw_para)^2)^2
        bw_weight[bw_para+1-i] = 15/16*(1-(trend[bw_para+1-i]/bw_para)^2)^2
    end
    bw_weight = bw_weight./sum(bw_weight)
    return bw_weight, trend
end
function gain(h::AbstractVector, w::Real)
    # Calculate Gain of filter h at Frequency w
    z = exp(-w*im)
    h1 = h[1]
    z1 = 1
    for i = 2:length(h)
        z1 = z1*z
        h1 = h1 + h[i]*z1
    end
    g2=h1*h1'
    fgain = sqrt(g2)
    return fgain
end

function bai_ng_criterion(m::DFMModel)
    fes = m.fes
    nbar = fes.nobs/fes.T # average observation per period
    g = log(min(nbar, fes.T))*(nbar+fes.T)/fes.nobs
    bn_icp = log(fes.ssr/fes.nobs)+ m.nfac_t*g
    return bn_icp
end
"""
- `nfac_max::Integer`: maximum number of factors
"""
struct FactorNumberEstimateStats
    bn_icp
    ssr_static
    R2_static
    aw_icp
    ssr_dynamic
    R2_dynamic
    tss::Float64
    nobs::Int
    T::Int
end
"""
- `m::DFMModel`: `DFMModel` specifying the model except number of unobservable
                 factors.
"""
function estimate_factor_numbers(m::DFMModel, nfacs::Union{Real, AbstractVector})
    max_nfac = maximum(nfacs)
    bn_icp = Vector{Union{Missing, Float64}}(undef, max_nfac)
    ssr_static = Vector{Float64}(undef, max_nfac)
    R2_static = Matrix{Union{Missing, Float64}}(undef, m.fes.ns, max_nfac)
    aw_icp = Matrix{Union{Missing, Float64}}(undef, max_nfac, max_nfac)
    ssr_dynamic = Matrix{Union{Missing, Float64}}(undef, max_nfac, max_nfac)
    R2_dynamic = Array{Union{Missing, Float64}}(undef, m.fes.ns, max_nfac, max_nfac)

    global tss, nobs, T
    for (i, nfac) = enumerate(1:max_nfac)
        dfmm = DFMModel(m.data, m.inclcode,
                m.nt_min_factor_estimation, m.nt_min_factorloading_estimation,
                m.initperiod, m.lastperiod, m.nfac_o, nfac, m.tol, m.n_uarlag, m.n_factorlag)
        estimate_factor!(dfmm)
        bn_icp[i] = bai_ng_criterion(dfmm)
        ssr_static[i] = dfmm.fes.ssr
        R2_static[:, i] = dfmm.fes.R2
        aw_icp[1:nfac, i], ssr_dynamic[1:nfac, i], R2_dynamic[:, 1:nfac, i] =
            amengual_watson_test(dfmm, 4)
        global tss = dfmm.fes.tss
        global nobs = dfmm.fes.nobs
        global T = dfmm.fes.T
    end
    return FactorNumberEstimateStats(bn_icp, ssr_static, R2_static,
                                     aw_icp, ssr_dynamic, R2_dynamic,
                                     tss, nobs, T)
end

function amengual_watson_test(m::DFMModel, nper::Integer)
    factor = m.factor
    T, ns, nfac_static = m.T, m.fes.ns, m.nfac_t
    nvar_lag = m.factor_var_model.nlag
    est_data = m.data[:, m.inclcode.==1]

    # Construct lags of factors and residuals for est_data
    x = [ones(T) lagmat(factor, 1:nvar_lag)]
    est_data_res = Array{Union{Missing, Float64}}(undef, T, ns)
    for is = 1:ns
        tmp, usedrows = drop_missing_row([est_data[:, is] x])
        y = tmp[:, 1]
        z = tmp[:, 2:end]
        ndf = size(z, 1)-size(z, 2)
        if ndf >= m.nt_min_factor_estimation  # Minimum degrees of freedom for series
            b, e = ols(y, z)
            est_data_res[findall(vec(usedrows)), is] = e
        end
    end

    # Carry out calculations for number of dynamic factors
    ssr = Array{Float64}(undef, nfac_static)
    r2  = Array{Union{Missing, Float64}}(undef, ns, nfac_static)
    aw  = Array{Float64}(undef, nfac_static)
    for nfac = 1:nfac_static
        dfmm = DFMModel(est_data_res, ones(count(m.inclcode.==1)),
                m.nt_min_factor_estimation, m.nt_min_factorloading_estimation,
                m.initperiod+4, m.lastperiod, 0, nfac, m.tol, m.n_uarlag, m.n_factorlag)
        estimate_factor!(dfmm)
        aw[nfac] = bai_ng_criterion(dfmm)
        ssr[nfac] = dfmm.fes.ssr
        r2[:, nfac] = dfmm.fes.R2
    end
    return aw, ssr, r2
end
