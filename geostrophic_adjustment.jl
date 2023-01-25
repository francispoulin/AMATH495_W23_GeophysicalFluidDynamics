using Oceananigans
using Oceananigans.Models: ShallowWaterModel

Lx, Ly, Lz = 10, 10, 1
Nx, Ny = 128, 128

grid = RectilinearGrid(size = (Nx, Ny),
                       x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                       topology = (Periodic, Bounded, Flat))

model = ShallowWaterModel(; grid, 
                          coriolis = FPlane(f=1), 
                          gravitational_acceleration = 1,
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO())

hⁱ(x, y, z)  = Lz + 0.1*exp(- x^2 - (y - 0*Ly/2)^2)
set!(model, h = hⁱ)

uh, vh, h = model.solution 

simulation = Simulation(model, Δt = 1e-2, stop_time = 20)

fields_filename = joinpath(@__DIR__, "geostrophic_adjustment_fields.nc")
simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; h),
                                                        filename = fields_filename,
                                                        schedule = TimeInterval(0.1),
                                                        overwrite_existing = true)

run!(simulation)

using NCDatasets, Printf, CairoMakie
x, y = xnodes(h), ynodes(h)
fig = Figure(resolution = (600, 600))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = AxisAspect(1),
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

ax_h  = Axis(fig[1, 1]; title = "Height, h", axis_kwargs...)

n = Observable(1)
ds = NCDataset(simulation.output_writers[:fields].filepath, "r")
times = ds["time"][:]
h = @lift ds["h"][:, :, 1, $n]
hm_h = heatmap!(ax_h, x, y, h, colorrange = (0.98, 1.025))
Colorbar(fig[1, 2], hm_h)
title = @lift @sprintf("t = %.1f", times[$n])

frames = 1:length(times)
record(fig, "geostrophic_adjustment.mp4", frames, framerate=12) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
close(ds)