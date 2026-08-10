using HDF5
include("StateRendering.jl")
using .StateRendering
estruct = h5open("src/Basic.h5")
CI = h5open("src/spectrum_results copy.h5")
 
display(StateRendering.PlotH5(estruct, CI))

