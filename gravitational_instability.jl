using Oceananigans
using CairoMakie, JLD2, Printf

Δt, time_end = 0.002, 20.0     
grid = RectilinearGrid(CPU(),
                      size=(128, 128), halo=(3, 3),
                      x = (0, 1),
                      z = (0, 1),
                      topology=(Periodic, Flat, Bounded))

const N² = -1.0   # Stratification        
const α  =  0.0   # Shear
const f  =  0.0   # Rotation

# Background Fields
B(x, y, z, t) = N² .* z
#V(x, y, z, t) = α .* x

# Initial Fields
uᵢ(x, y, z) = 0.01*rand()
bᵢ(x, y, z) = 0.0001*rand()

model = NonhydrostaticModel(; grid,
			      background_fields = (b=B,),           # v = V
                              coriolis = FPlane(f),
                              buoyancy = BuoyancyTracer(),
                              advection = WENO(),
                              tracers = (:b,))
set!(model, u = uᵢ, b = bᵢ)
simulation = Simulation(model, Δt=Δt, stop_time=time_end)
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end
simulation.callbacks[:print_progress] = Callback(print_progress, TimeInterval(0.1))

u, v, w = model.velocities
b = model.tracers.b
outputs = (; u, w, b)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs;
                                                  filename = "gravitational_instability",
                                                  schedule = TimeInterval(0.1),
                                        overwrite_existing = true)
run!(simulation)

filename = "gravitational_instability.jld2"
file=jldopen(filename)

xw, yw, zw = nodes((Center, Center, Face), grid)
xb, yb, zb = nodes((Center, Center, Center), grid)

### Make animation
iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
w_field = @lift file["timeseries/w/" * string($iter)][:, 1, :]
b_field = @lift file["timeseries/b/" * string($iter)][:, 1, :]
fig = Figure(resolution = (2000, 1000))
title_w = @lift(@sprintf("w at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
title_b = @lift(@sprintf("b at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
ax_w = Axis(fig[1,1], xlabel = "x", ylabel = "z", title=title_w)
ax_b = Axis(fig[1,3], xlabel = "x", ylabel = "z", title=title_b)
heatmap_w = heatmap!(ax_w, xw, zw, w_field, colormap=:balance)
heatmap_b = heatmap!(ax_b, xb, zb, b_field, colormap=:balance)
Colorbar(fig[1,2], heatmap_w, label="w", width=25)
Colorbar(fig[1,4], heatmap_b, label="b", width=25)
display(fig)

output_prefix = "gravitational_instability"
record(fig, output_prefix * ".mp4", iters[2:end], framerate=6) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end