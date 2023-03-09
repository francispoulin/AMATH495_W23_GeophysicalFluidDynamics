using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode
using CairoMakie, JLD2, Printf

Nx, Nz = 200, 200         
Lx, Lz = 100kilometers, 1kilometers  
Δt, stop_time = 30, 6hours

coriolis = FPlane(f=1e-3)

grid = RectilinearGrid(CPU();
                       size = (Nx, Nz), 
                          x = (-Lx/2, Lx/2),
                          z = (-Lz, 0),
                   topology = (Periodic, Flat, Bounded))                   

model = NonhydrostaticModel(; grid,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            coriolis = coriolis,
                            closure = AnisotropicMinimumDissipation()
)

N = 1e-2
bᵢ(x, y, z) = N^2 * z + 0.01 * exp(-200*x^2 / Lx^2) * exp(-40*(z + Lz/2)^2 / Lz^2)

set!(model, b=bᵢ)
simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|b|) = %.1e C, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.tracers.b), 
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(2minutes))
simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers), 
                                                      schedule = TimeInterval(2minutes),
                                                       filename="stratified_geostrophic_adjustment.jld2", 
                                            overwrite_existing = true)

run!(simulation)

filename = "stratified_geostrophic_adjustment.jld2"
file=jldopen(filename)

xw, yw, zw = nodes((Center, Center, Face), grid)
xb, yb, zb = nodes((Center, Center, Center), grid)

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
w_field = @lift file["timeseries/w/" * string($iter)][:, 1, :]
b_field = @lift file["timeseries/b/" * string($iter)][:, 1, :]
fig = Figure(resolution = (2000, 1000))
title_w = @lift(@sprintf("w at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
title_b = @lift(@sprintf("b at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
ax_w = Axis(fig[1,1], xlabel = "x (km)", ylabel = "z (km)", title=title_w)
ax_b = Axis(fig[2,1], xlabel = "x (km)", ylabel = "z (km)", title=title_b)
heatmap_w = heatmap!(ax_w, xw/kilometers, zw/kilometers, w_field, colormap=:balance)
heatmap_b = heatmap!(ax_b, xb/kilometers, zb/kilometers, b_field, colormap=:balance)
Colorbar(fig[1,2], heatmap_w, label="w", width=25)
Colorbar(fig[2,2], heatmap_b, label="b", width=25)
display(fig)

output_prefix = "stratified_geostrophic_adjustment"
record(fig, output_prefix * ".mp4", iters[2:end], framerate=6) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end

close(file)
