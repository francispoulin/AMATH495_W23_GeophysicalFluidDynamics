using Oceananigans, CairoMakie, Printf

grid = RectilinearGrid(size=(256, 128), x=(-10, 10), z=(-5, 5),
                       topology=(Periodic, Flat, Bounded))

shear_flow(x, y, z, t) = tanh(z)
stratification(x, y, z, t, p) = p.h * p.Ri * tanh(z / p.h)

U, B = BackgroundField(shear_flow), BackgroundField(stratification, parameters=(Ri=0.1, h=1/4)) 

zF, zC = znodes(Face, grid), znodes(Center, grid)

Ri, h = B.parameters

### Plot Profiles
fig = Figure(resolution = (850, 450))
ax = Axis(fig[1, 1], xlabel = "U(z)", ylabel = "z")
lines!(ax, shear_flow.(0, 0, zC, 0), zC; linewidth = 3)
ax = Axis(fig[1, 2], xlabel = "B(z)")
lines!(ax, [stratification(0, 0, z, 0, (Ri=Ri, h=h)) for z in zC], zC; linewidth = 3, color = :red)
ax = Axis(fig[1, 3], xlabel = "Ri(z)")
lines!(ax, [Ri * sech(z / h)^2 / sech(z)^2 for z in zF], zF; linewidth = 3, color = :black)
save("profiles.png", fig)

model = NonhydrostaticModel(timestepper = :RungeKutta3,
                              advection = UpwindBiasedFifthOrder(),
                                   grid = grid,
                               coriolis = nothing,
                      background_fields = (u=U, b=B),
                                closure = ScalarDiffusivity(ν=2e-4, κ=2e-4),
                               buoyancy = BuoyancyTracer(),
                                tracers = :b)

simulation = Simulation(model, Δt=0.05, stop_iteration=2500)

u, v, w = model.velocities
b = model.tracers.b

perturbation_vorticity = Field(∂z(u) - ∂x(w))

xω, yω, zω = nodes(perturbation_vorticity)
xb, yb, zb = nodes(b)

total_vorticity = Field(∂z(u) + ∂z(model.background_fields.velocities.u) - ∂x(w))
total_b = Field(b + model.background_fields.tracers.b)

simulation.output_writers[:vorticity] =
    JLD2OutputWriter(model, (ω=perturbation_vorticity, Ω=total_vorticity, b=b, B=total_b),
                     schedule = TimeInterval(0.5),
                     filename = "kelvin_helmholtz_instability.jld2",
                     overwrite_existing = true)

noise(x, y, z) = 0.001*randn()
set!(model, u=noise)
run!(simulation)

### Make animation
filepath = simulation.output_writers[:vorticity].filepath
ω_timeseries = FieldTimeSeries(filepath, "ω")
b_timeseries = FieldTimeSeries(filepath, "b")
Ω_timeseries = FieldTimeSeries(filepath, "Ω")
B_timeseries = FieldTimeSeries(filepath, "B")
times = ω_timeseries.times
frames = 1:length(times)
t_final = times[end]

n = Observable(1)
Ωₙ = @lift interior(Ω_timeseries, :, 1, :, $n)
Bₙ = @lift interior(B_timeseries, :, 1, :, $n)
fig = Figure(resolution=(1600, 600))
kwargs = (xlabel="x", ylabel="z", limits = ((xω[1], xω[end]), (zω[1], zω[end])), aspect=1,)
title = @lift @sprintf("t = %.2f", times[$n])
ax_Ω = Axis(fig[2, 1]; title = "total vorticity", kwargs...)
ax_B = Axis(fig[2, 3]; title = "total buoyancy", kwargs...)
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)
hm_Ω = heatmap!(ax_Ω, xω, zω, Ωₙ; colorrange = (-1, 1), colormap = :balance)
Colorbar(fig[2, 2], hm_Ω)
hm_B = heatmap!(ax_B, xb, zb, Bₙ; colorrange = (-0.04, 0.04), colormap = :balance)
Colorbar(fig[2, 4], hm_B)
record(fig, "kelvin_helmholtz_instability_total.mp4", frames, framerate=8) do i
       @info "Plotting frame $i of $(frames[end])..."
       n[] = i
end
