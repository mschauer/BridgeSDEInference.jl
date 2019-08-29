using Bridge
using Random
using DataFrames
using CSV

SRC_DIR = joinpath(Base.source_dir(), "..", "src")
AUX_DIR = joinpath(SRC_DIR, "auxiliary")
include(joinpath(AUX_DIR, "data_simulation_fns.jl"))
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "JRNeural.jl"))
FILENAME_OUT = joinpath(OUT_DIR,
                     "jr_path_part_obs.csv")

### parameters as Table 1  of
# https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0046-4
P = JRNeuralDiffusion(3.25, 0.1, 22.0, 0.05 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0, 0.01 , 2000.0, 1.0)
# starting point under :regular parametrisation
x0 = ℝ{6}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

dt = 1/50000
T = 200.0
tt = 0.0:dt:T

Random.seed!(4)
XX, _ = simulateSegment(ℝ{3}(0.0, 0.0, 0.0), x0, P, tt)
#weird values

XX.yy
num_obs = 100
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
#observation x[2]- x[3]
df = DataFrame(time=Time, x1=[x[2] - x[3] for x in XX.yy[1:skip:end]])
CSV.write(FILENAME_OUT, df)