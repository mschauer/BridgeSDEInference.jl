#inference without blocking for Jansen and Rit model.
#for the editing and reading the notes about the porcess and some derivation
#see https://www.overleaf.com/2487461149mmywgchrdbgr


SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "BridgeSDEInference.jl"))
using Main.BridgeSDEInference
using Distributions # to define priors
using Random        # to seed the random number generator
using DataFrames
using CSV
include(joinpath(AUX_DIR, "read_and_write_data.jl"))
include(joinpath(AUX_DIR, "transforms.jl"))

# decide if first passage time observations or partially observed diffusion
fptObsFlag = false

# pick dataset
filename = "jr_path_part_obs.csv"

# fetch the data
(df, x0, obs, obsTime, fpt,
      fptOrPartObs) = readData(Val(fptObsFlag), joinpath(OUT_DIR, filename))


# Initial parameter guess.

θ₀ = [3.25, 0.1, 22.0, 0.05 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0, 0.01 , 2000.0, 1.0]

# Target law
P˟ = JRNeuralDiffusion(θ₀...)

# Auxiliary law
P̃ = [JRNeuralDiffusionAux2(θ₀..., t₀, u[1], T, v[1]) for (t₀,T,u,v)
     in zip(obsTime[1:end-1], obsTime[2:end], obs[1:end-1], obs[2:end])]

display(P̃[1])


L = @SMatrix [0. 1. -1. 0. 0. 0.]
Σdiagel = 10^(-10)
Σ = @SMatrix [Σdiagel]

Ls = [L for _ in P̃]
Σs = [Σ for _ in P̃]
τ(t₀,T) = (x) ->  t₀ + (x-t₀) * (2-(x-t₀)/(T-t₀))
numSteps=1*10^5
saveIter=3*10^2

## For σ_y (positive), μ_y, C (positive), b (positive)
tKernel = RandomWalk([5.0, 0.0, 5.0, 5.0],
               [true, false, true, true])

## Automatic assignment of indecesForUpdt
priors = Priors((ImproperPrior(), Normal(0.0, 100.0), ImproperPrior(),  ImproperPrior()))


𝔅 = NoBlocking()
blockingParams = ([], 0.1, NoChangePt())

changePt = NoChangePt()

#x0Pr = KnownStartingPt(x0)

x0Pr = GsnStartingPt(x0, x0, @SMatrix [20. 0 0 0 0 0;
                                    0 20. 0 0 0 0;
                                    0 0 20. 0 0 0;
                                    0 0 0 20. 0 0;
                                    0 0 0 0 20. 0;
                                    0 0 0 0 0 20.])

warmUp = 100
Random.seed!(4)
start = time()
(chain, accRateImp, accRateUpdt,
    paths, time_) = mcmc(eltype(x0), fptOrPartObs, obs, obsTime, x0Pr, 0.0, P˟,
                         P̃, Ls, Σs, numSteps, tKernel, priors, τ;
                         fpt=fpt,
                         ρ=0.975,
                         dt=1/1000,
                         saveIter=saveIter,
                         verbIter=10^2,
                         #TOCHANGE
                         updtCoord=(Val((true, false, false, false)),
                                    Val((false, true, false, false)),
                                    Val((false, false, true, false)),
                                    Val((false, false, false, true)),
                                    ),
                         paramUpdt=true,
                         updtType=(MetropolisHastingsUpdt(),
                                    ConjugateUpdt(),
                                    MetropolisHastingsUpdt(),
                                    MetropolisHastingsUpdt(),
                                    ),
                         skipForSave=10^0,
                         blocking=𝔅,
                         blockingParams=blockingParams,
                         solver=Vern7(),
                         changePt=changePt,
                         warmUp=warmUp)
elapsed = time() - start
print("time elapsed: ", elapsed, "\n")

print("imputation acceptance rate: ", accRateImp,
      ", parameter update acceptance rate: ", accRateUpdt)

x0⁺, pathsToSave = transformMCMCOutput(x0, paths, saveIter; chain=chain,
                                       numGibbsSteps=2,
                                       parametrisation=param,
                                       warmUp=warmUp)


df2 = savePathsToFile(pathsToSave, time_, joinpath(OUT_DIR, "jr_sampled_paths.csv"))
df3 = saveChainToFile(chain, joinpath(OUT_DIR, "jr_chain.csv"))

include(joinpath(AUX_DIR, "plotting_fns.jl"))
set_default_plot_size(30cm, 20cm)
plotPaths(df2, obs=[Float64.(df.x1), [x0⁺[2]]],
          obsTime=[Float64.(df.time), [0.0]], obsCoords=[1,2])

plotChain(df3, coords=[1])
plotChain(df3, coords=[2])
