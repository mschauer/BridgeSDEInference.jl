
using Bridge
using StaticArrays
import Bridge: b, σ, B, β, a, constdiff
const ℝ = SVector{N, T} where {N, T}

"""
    JRNeuralDiffusion <: ContinuousTimeProcess{ℝ{6, T}}

structure defining the Jansen and Rit Neural Mass Model described in
https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0046-4 and
https://arxiv.org/abs/1903.01138
"""
struct JRNeuralDiffusion{T} <: ContinuousTimeProcess{ℝ{6, T}}
    A::T
    a::T
    B::T
    b::T
    C1::T
    C2::T
    C3::T
    C4::T
    νmax::T
    v0::T
    r::T
    μx::T
    μy::T
    μz::T
    σx::T
    σy::T
    σz::T
    # default constructor
    function JRNeuralDiffusion(A::T, a::T, B::T, b::T, C1::T,  C2::T,  C3::T,  C4::T,
            νmax::T, v0::T ,r::T, μx::T, μy::T, μz::T, σx::T, σy::T, σz::T) where T
        new{T}(A, a, B, b, C1, C2, C3, C4, νmax, v0, r, μx, μy, μz, σx, σy, σz)
    end
    # constructor given assumption statistical paper
    function JRNeuralDiffusion(A::T, a::T, B::T, b::T, C::T,
            νmax::T, v0::T ,r::T, μx::T, μy::T, μz::T, σx::T, σy::T, σz::T) where T
        new{T}(A, a, B, b, C, 0.8C, 0.25C, 0.25C, νmax, v0, r, μx, μy, μz, σx, σy, σz)
    end
end

# in the statistical paper they set μ's to be constant and not function of time.
function μx(t, P::JRNeuralDiffusion{T}) where T
    P.μx
end

function μy(t, P::JRNeuralDiffusion{T}) where T
    P.μy
end

function μz(t, P::JRNeuralDiffusion{T}) where T
    P.μz
end

"""
    sigm(x, P::JRNeuralDiffusion)

definition of sigmoid function
"""
function sigm(x, P::JRNeuralDiffusion{T}) where T
    P.νmax / (1 + exp(P.r*(P.v0 - x)))
end


function b(t, x, P::JRNeuralDiffusion{T}) where T
    ℝ{6}(x[4], x[5], x[6],
    P.A*P.a*(μx(t, P) + sigm(x[2] - x[3], P)) - 2P.a*x[4] - P.a*P.a*x[1],
    P.A*P.a*(μy(t, P) + P.C2*sigm(P.C1*x[1], P)) - 2P.a*x[5] - P.a*P.a*x[2],
    P.B*P.b*(μz(t, P) + P.C4*sigm(P.C3*x[1], P)) - 2P.b*x[6] - P.b*P.b*x[3])
end

#6X3 matrix
function σ(t, x, P::JRNeuralDiffusion{T}) where T
    @SMatrix    [0.0  0.0  0.0;
                0.0  0.0  0.0;
                0.0  0.0  0.0;
                P.σx  0.0  0.0;
                0.0  P.σy  0.0;
                0.0  0.0  P.σz;]
end

constdiff(::JRNeuralDiffusion) = true
clone(::JRNeuralDiffusion, θ) = JRNeuralDiffusion(θ...)
params(P::JRNeuralDiffusion) = [P.A, P.a, P.B, P.b, P.C1, P.C2, P.C3, P.C4, P.νmax,
    P.v0, P.r, P.μx, P.μy, P.μ_z, P.σx, P.σy, P.σz]


#### Conjugate posterior for μ_y  ###
#drift as b^[2](t, x) = μ*ϕ + R(t, x)
#vector in ℝ³
phi(::Val{0}, t, x, P::JRNeuralDiffusion) = @SVector [P.A*P.a*(P.μx + sigm(x[2] - x[3], P)) - 2P.a x[4] - P.a*P.a*x[1],
                                P.A*P.a*P.C2*sigm(P.C1*x[1], P) - 2P.a x[5] - P.a*P.a*x[2],
                                P.B*P.b*(P.μz + P.C4*sigm(P.C3*x[1], P)) - 2P.b x[6] - P.b*P.b*x[3] ]
#vector in ℝ³
phi(::Val{1}, t, x, P::JRNeuralDiffusion) = @SVector  [0, P.A*P.a0, 0]
phi(::Val{2}, t, x, P::JRNeuralDiffusion) = @SVector [0.0, 0.0, 0.0]
phi(::Val{3}, t, x, P::JRNeuralDiffusion) = @SVector [0.0, 0.0, 0.0]



"""
    JRNeuralDiffusionaAux1{T, S1, S2} <: ContinuousTimeProcess{ℝ{6, T}}

structure for the auxiliary process (defined as linearized process in the final point)
"""
struct JRNeuralDiffusionAux1{R, S1, S2} <: ContinuousTimeProcess{ℝ{6, R}}
    A::R
    a::R
    B::R
    b::R
    C1::R
    C2::R
    C3::R
    C4::R
    νmax::R
    v0::R
    r::R
    μx::R
    μy::R
    μz::R
    σx::R
    σy::R
    σz::R
    u::S1
    t::Float64
    v::S2
    T::Float64
    # default generator
    function JRNeuralDiffusionAux1(A::R, a::R, B::R, b::R, C1::R, C2::R, C3::R, C4::R,
                        νmax::R, v0::R , r::R, μx::R, μy::R, μz::R, σx::R,
                         σy::R, σz::R, t::Float64, u::S1, T::Float64, v::S2) where {R, S1, S2}
        new{R, S1, S2}(A, a, B, b, C1, C2, C3, C4, νmax, v0, r, μx, μy, μz, σx, σy, σz, t, u, T, v)
    end

    # generator given assumptions paper
    function JRNeuralDiffusionAux1(A::R, a::R, B::R, b::R, C::R,
                        νmax::R, v0::R ,r::R, σx::R, σy::R, σz::R, t::Float64, u::S1,
                        T::Float64, v::S2) where {R, S1, S2}
        new{R, S1, S2}(A, a, B, b, C, 0.8C, 0.25C, 0.25C, νmax, v0, r, σx, σy, σz, t, u, T, v)
    end
end

"""
    d1sigm(x, P::JRNeuralDiffusionAux1{T, S1, S2})

derivative of sigmoid function
"""
function d1sigm(x, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    P.νmax*r*exp(r*(v0 - x))/(1 + exp(r*(v0 - x)))^2
end

function B(t, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    @SMatrix [0.0  0.0  0.0  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  1.0  0.0;
              0.0  0.0  0.0  0.0  0.0  1.0;
              -P.a*P.a  P.A*P.a*d1sigm(P.v[2] - P.v[3], P)  -P.A*P.a*d1sigm(P.v[2] - P.v[3], P)   -2P.a  0.0  0.0;
              P.A*P.a*P.C1*P.C2*d1sigm(P.C1*P.v[1], P)  -P.a*P.a  0.0  0.0  -2P.a  0.0;
              P.B*P.b*P.C3*P.C4*d1sigm(P.C3*P.v[1], P)  0.0  -P.b*P.b  0.0  0.0  -2P.b]
end


function β(t, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    ℝ{6}(0.0, 0.0, 0.0,
        P.A*P.a*(μx(t, P) + sigm(P.v[2] - P.v[3], P) - d1sigm(P.v[2] - P.v[3], P)*(P.v[2] - P.v[3])),
        P.A*P.a*(μy(t, P) + P.C2*(sigm(P.C1*P.v[1], P) - d1sigm(P.C1*P.v[1], P)*(P.C1*P.v[1]))),
        P.B*P.b*(μz(t, P) + P.C4*(sigm(P.C3*P.v[1], P) - d1sigm(P.C3*P.v[1], P)*(P.C3*P.v[1]))) )
end

function σ(t, P::JRNeuralDiffusionAux1{T, S1, S2}) where {T, S1, S2}
    @SMatrix    [0.0  0.0  0.0;
                0.0  0.0  0.0;
                0.0  0.0  0.0;
                P.σx  0.0  0.0;
                0.0  P.σy  0.0;
                0.0  0.0  P.σz;]
end



b(t, x, P::JRNeuralDiffusionAux1) = B(t,P) * x + β(t,P)
a(t, P::JRNeuralDiffusionAux1) = σ(t,P) * σ(t, P)'

clone(P::JRNeuralDiffusionAux1, θ) = JRNeuralDiffusionAux1(θ..., P.t,
                                                         P.u, P.T, P.v)

clone(P::JRNeuralDiffusionAux1, θ, v) = JRNeuralDiffusionAux1(θ..., P.t,
                                                            zero(v), P.T, v)

params(P::JRNeuralDiffusionAux1) = [P.A, P.a, P.B, P.b, P.C1, P.C2, P.C3, P.C4, P.νmax,
    P.v0, P.r, P.μ_x, P.μ_y, P.μ_z, P.σx, P.σy, P.σz]


"""
    JRNeuralDiffusionaAux2{T, S1, S2} <: ContinuousTimeProcess{ℝ{6, T}}

structure for the auxiliary process defined as linearized process in the final point
for the random variable V_t = LX_t and around the point tt in ℝ¹ (user choice, if not specified around v0) for the unobserved first components.
the final point v should be a float number or an array?
"""
struct JRNeuralDiffusionAux2{R, S1, S2} <: ContinuousTimeProcess{ℝ{6, R}}
    tt::R
    A::R
    a::R
    B::R
    b::R
    C1::R
    C2::R
    C3::R
    C4::R
    νmax::R
    v0::R
    r::R
    μx::R
    μy::R
    μz::R
    σx::R
    σy::R
    σz::R
    u::S1
    t::Float64
    v::S2
    T::Float64
    # default generator
    function JRNeuralDiffusionAux2(A::R, a::R, B::R, b::R, C1::R, C2::R, C3::R, C4::R,
                        νmax::R, v0::R , r::R, μx::R, μy::R, μz::R, σx::R,
                         σy::R, σz::R, t::Float64, u::S1, T::Float64, v::S2; tt = v0) where {R, S1, S2}
        new{R, S1, S2}(tt, A, a, B, b, C1, C2, C3, C4, νmax, v0, r, μx, μy, μz, σx, σy, σz, t, u, T, v)
    end

    # generator given assumptions paper
    function JRNeuralDiffusionAux2(A::R, a::R, B::R, b::R, C::R,
                        νmax::R, v0::R ,r::R, σx::R, σy::R, σz::R, t::Float64, u::S1,
                        T::Float64, v::S2; tt = v0) where {R, S1, S2}
        new{R, S1, S2}(A, a, B, b, C, 0.8C, 0.25C, 0.25C, νmax, v0, r, σx, σy, σz, t, u, T, v)
    end
end

"""
    d1sigm(x, P::JRNeuralDiffusionAux2{T, S1, S2})

derivative of sigmoid function
"""
function d1sigm(x, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    P.νmax*r*exp(r*(v0 - x))/(1 + exp(r*(v0 - x)))^2
end


function B(t, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    @SMatrix [0.0  0.0  0.0  1.0  0.0  0.0;
              0.0  0.0  0.0  0.0  1.0  0.0;
              0.0  0.0  0.0  0.0  0.0  1.0;
              -P.a*P.a  P.A*P.a*d1sigm(P.v[1], P)  -P.A*P.a*d1sigm(P.v[1], P)   -2P.a  0.0  0.0;
              P.A*P.a*P.C1*P.C2*d1sigm(P.C1*P.tt, P)  -P.a*P.a  0.0  0.0  -2P.a  0.0;
              P.B*P.b*P.C3*P.C4*d1sigm(P.C3*P.tt, P)  0.0  -P.b*P.b  0.0  0.0  -2P.b]
end


function β(t, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    ℝ{6}(0.0, 0.0, 0.0,
        P.A*P.a*(μx(t, P) + sigm(P.v[1], P) - d1sigm(P.v[1], P)*(P.v[1])),
        P.A*P.a*(μy(t, P) + P.C2*(sigm(P.C1*P.tt[1], P) - d1sigm(P.C1*P.tt[1], P)*(P.C1*P.tt[1]))),
        P.B*P.b*(μz(t, P) + P.C4*(sigm(P.C3*P.tt[1], P) - d1sigm(P.C3*P.tt[1], P)*(P.C3*P.tt[1]))) )
end

function σ(t, P::JRNeuralDiffusionAux2{T, S1, S2}) where {T, S1, S2}
    @SMatrix    [0.0  0.0  0.0;
                0.0  0.0  0.0;
                0.0  0.0  0.0;
                P.σx  0.0  0.0;
                0.0  P.σy  0.0;
                0.0  0.0  P.σz;]
end


b(t, x, P::JRNeuralDiffusionAux2) = B(t,P) * x + β(t,P)
a(t, P::JRNeuralDiffusionAux2) = σ(t,P) * σ(t, P)'
clone(P::JRNeuralDiffusionAux2, θ) = JRNeuralDiffusionAux(θ..., P.t, P.u, P.T, P.v)
clone(P::JRNeuralDiffusionAux2, θ, v) = JRNeuralDiffusionAux(θ..., P.t, zero(v), P.T, v)
params(P::JRNeuralDiffusionAux2) = [P.A, P.a, P.B, P.b, P.C1, P.C2, P.C3, P.C4, P.νmax,
    P.v0, P.r, P.μ_x, P.μ_y, P.μ_z, P.σx, P.σy, P.σz]