
# using DataFrames
using Statistics
using DelimitedFiles
using LinearAlgebra
using Plots
gr()
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
struct DFMModel
    data
    inclcode
    T::Int
    nt_min_factor_estimation::Int
    nt_min_factorloading_estimation::Int
    initperiod::Int
    lastperiod::Int
    nfac_o::Int
    nfac_u::Int
    nfac_t::Int
    tol::Float64
    factor
    lambda
    uar_coef
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

    size(data, 2) == size(inclcode, 2) || error("data and inclcode must have same column size")
    initperiod < lastperiod || error("initperiod must be smaller than lastperiod")
    ((n_uarlag > 0) && (n_factorlag > 0)) || error("n_uarlag and n_factorlag must be positive")

    T, ns = size(data)
    nfac_t = nfac_o+nfac_u
    factor = Matrix{Union{Missing, Float64}}(missing, T, nfac_t)
    lambda = Matrix{Float64}(undef, ns, nfac_t)
    uar_coef = Matrix{Float64}(undef, ns, n_uarlag)
    uar_ser = Vector{Float64}(undef, ns)
    factor_var_model = VARModel(factor, n_factorlag, initperiod= initperiod,
                                lastperiod=lastperiod)
    return DFMModel(data, inclcode, T,
                    nt_min_factor_estimation, nt_min_factorloading_estimation,
                    initperiod, lastperiod, nfac_o, nfac_u, nfac_t,
                    tol, factor, lambda, uar_coef, uar_ser,
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
function estimate_factor!(m, xbal, xdatastd, xdata_standardized)
    data = m.data
    initperiod, lastperiod, nt_min, nfac_u, nfac_o, tol =
        m.initperiod, m.lastperiod, m.nt_min_factor_estimation,
        m.nfac_u, m.nfac_o, m.tol
    nt, ns = size(xdata_standardized)
    # Get initial F_t given Lambda_t using PCA
    f = pca_score(xbal, nfac_u)
    ssr = 0
    diff = 1000
    while diff > tol*nt*ns
        ssr_old = ssr
        # given F_t, get Lambda_t
        if size(xdata_standardized, 1) >= nt_min # if sample size is enough
            lambda = ols_skipmissing(xdata_standardized, f, Unbalanced())[1]'
        end
        # given Lambda_t, get F_t by regressing X_t on Lambda_t for each t
        tmp = ols_skipmissing(xdata_standardized', lambda[:, nfac_o+1:end], Unbalanced())
        f, ehat = tmp[1]', tmp[2]
        ssr = sum(sum(skipmissing(ehat.^2)))
        diff = abs(ssr_old - ssr)
        println("diff = ", diff)
    end
    m.factor[initperiod:lastperiod,  :] = f
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
            ssr = sum(uhat.^2)
            ym = y_used .- mean(y_used)
            tss = sum(ym.^2)
            r2_mat[is] = 1-ssr/tss
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
    datamean = [mean(skipmissing(data[:, i])) for i = 1:size(data, 2)]'
    # # make correction (which I don't understand why being needed)
    tmp = size(data, 1) .- sum(ismissing.(data), dims = 1)
    tmp = (tmp.-1)./tmp
    datastd = [std(skipmissing(data[:, i])) for i = 1:size(data, 2)]'.*sqrt.(tmp)
    data_standardized = (data .- datamean)./datastd
    return data_standardized, datastd
end

function estimate!(m::DFMModel, ::NonParametric=NonParametric())
    # use part of the data
    est_data = m.data'[vec(m.inclcode.==1), :]'
    xdata = est_data[m.initperiod:m.lastperiod, :]

    # preprocess data to have unit standard error
    xdata_standardized, xdatastd = standardize_data(xdata)

    tss = sum(skipmissing(xdata_standardized.^2))

    # estimate factor by iteration using balanced data
    xbal, _ = drop_missing_col(xdata_standardized)
    estimate_factor!(m, xbal, xdatastd, xdata_standardized)

    # estimate factor loading using full sample
    estimate_factor_loading!(m)

    # estimate the equation of factor evolution
    estimate_var!(m.factor_var_model)
    return nothing
end
compute_series(dfmm::DFMModel, is::Integer) = dfmm.factor*dfmm.lambda[is, :]
detrended_year_growth(y::AbstractVector) = vec(sum(lagmat(y, 0:3), dims=2))

find_row_number(year, date) = findall(year.==date)[1]
function compute_r2(y::AbstractVector, e::AbstractVector)
    sse = dot(e, e)
    ssy = dot(y.-mean(y), y.-mean(y))
    return 1-(sse/ssy)
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
