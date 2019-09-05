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
P = JRNeuralDiffusion(3.25, 100.0, 22.0, 50.0 , 135.0, 5.0, 6.0, 0.56, 0.0, 220.0, 0.0,2000.0)
# starting point under :regular parametrisation



display(P)

x0 = ℝ{6}(0.08,18,15,-0.5,0,0)

dt = 1/100000
T = 10.0
tt = 0.0:dt:T

Random.seed!(4)
XX, _ = simulateSegment(ℝ{1}( 0.0), x0, P, tt)


XX.yy
num_obs = 1000
skip = div(length(tt), num_obs)
Time = collect(tt)[1:skip:end]
#observation x[2]- x[3]
df = DataFrame(time=Time, x1=[x[2] - x[3] for x in XX.yy[1:skip:end]])
CSV.write(FILENAME_OUT, df)


using Plots
plot(df.time, df.x1)
