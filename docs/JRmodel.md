## Application of BridgeSDEInference.jl for the Jansen and Rit Neural Mass model.
### References
Model: https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0046-4, equation 4.
ABC inference: https://arxiv.org/pdf/1903.01138.pdf, section 5
### Defining the model and observation scheme:
The non-linear hypo-elliptic 6-dimensional JRNM model is defined through the function
```julia
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
```
We observe discretly on time a linear combination V_t = L X_t where
```
L = @SMatrix [0. 1. -1. 0. 0. 0.]
```
### Defining the auxiliary model for the pulling term
For our purposes, we define 2 auxiliary models. One is the linearization of the original model at the final points (`JRNeuralDiffusionAux1`) and the second is the linearization of the original model for the difference the second and third dimension which we actually observe and a linearization around a point fixed by the user for the first dimension which we cannot observe directly (`JRNeuralDiffusionAux2`).

### Simulation
As already discussed in [this note](docs/generate_data.md), in the file [generate_data.md](docs/generate_data.md) we simulate the whole process and retain at discrete time some data according to the above mentioned observation scheme. The plan is to use real data.

### Statistical inference on some parameters
As discussed in https://arxiv.org/pdf/1903.01138.pdf, subsection 5.2.1., there is a identifiability issue. We fix some parameters and we perform inference to the triple `σy, μy, C`. In the mcmc algorithm we set up a  conjugate step for `μy` and a Metropolis Hasting step for `σy` and `C`.  

First we to define..... 


for the conjugate step we need to set the functions \phi which will be used to represent the drift `b(t, x) = \phi_0(t,x) +  μy \phi_1(t, x)`  and we set `\phi_2(t, x)`, `\phi_3(t, x)` to `0` since we do not need to use these functions for `C` and `σy`
```julia
phi(::Val{0}, t, x, P::JRNeuralDiffusion) = @SVector [P.A*P.a*(P.μx + sigm(x[2] - x[3], P)) - 2P.a x[4] - P.a*P.a*x[1],
                                P.A*P.a*P.C2*sigm(P.C1*x[1], P) - 2P.a x[5] - P.a*P.a*x[2],
                                P.B*P.b*(P.μz + P.C4*sigm(P.C3*x[1], P)) - 2P.b x[6] - P.b*P.b*x[3] ]
phi(::Val{1}, t, x, P::JRNeuralDiffusion) = @SVector  [0, P.A*P.a0, 0]
phi(::Val{2}, t, x, P::JRNeuralDiffusion) = @SVector [0.0, 0.0, 0.0]
phi(::Val{3}, t, x, P::JRNeuralDiffusion) = @SVector [0.0, 0.0, 0.0]
```

















