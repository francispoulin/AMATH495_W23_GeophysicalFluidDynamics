### Ekman Layer Surface

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, znode   
using CairoMakie
using JLD2
using Printf

## Parameters
const Lx = 1kilometers 
const Lz = 200 

Δt₀, stop_time = 10, 10days
Nx, Nz = 2, 64

ρ₀ = 1028
νz = 1e-2
parameters = (Lx = Lx,
              Lz = Lz,
              τ = 0.1 / (ρ₀*νz))
## Grid
σ = 1.2      # stretching factor
hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))
grid = RectilinearGrid(CPU();
                           size = (Nx, Nz),
                           halo = (3, 3),
                              x = (-Lx/2, Lx/2),
                              z = hyperbolically_spaced_faces,
                       topology = (Periodic, Flat, Bounded))


## Plot vertical grid
f  = Figure()
ax = Axis(f[1,1], title="Vertical Grid Spacing", xlabel="Δz", ylabel="z")
lines!(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz], linewidth=4, label="Δz")
scatter!(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz])
axislegend()
Makie.save("vertical-grid.png", f)

## Model
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(parameters.τ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0))
model = NonhydrostaticModel(
                   grid = grid,
              advection = WENO(grid = grid),
            #coriolis = FPlane(latitude=45),
               #coriolis = NonTraditionalBetaPlane(fy=1e-4, fz=1e-4, β = 0, γ = 0,),
               coriolis = NonTraditionalBetaPlane(latitude=45),
                closure = VerticalScalarDiffusivity(ν = νz),                                   
    boundary_conditions = (u=u_bcs,v=v_bcs,)
             )

τ = parameters.τ
f = model.coriolis.fz
d = sqrt(2*νz/f)
             
u_Ek(x, z) = -sqrt(2)/(f*d)*τ*exp.(z/d).*cos.(z/d .- π/4)
v_Ek(x, z) = -sqrt(2)/(f*d)*τ*exp.(z/d).*sin.(z/d .- π/4)

set!(model, u = u_Ek, v = v_Ek)

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, max(u): (%6.3e, %6.3e) m/s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v))

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1))

## Output
simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities,);
                                                      filename = "surface_Ekman_layer_fields",
                                                      schedule = TimeInterval(1hour),
                                                      indices = (:, :, :),
                                                      overwrite_existing = true)

run!(simulation)

## Movie

filename = "surface_Ekman_layer_fields.jld2"
file=jldopen(filename)

z = znodes(Center, grid)

u_Ekplt = -sqrt(2)/(f*d)*τ*exp.(z/d).*cos.(z/d .- π/4) #FJP: why?
v_Ekplt = -sqrt(2)/(f*d)*τ*exp.(z/d).*sin.(z/d .- π/4) #FJP: why?

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
u = @lift file["timeseries/u/" * string($iter)][1, 1, :]
v = @lift file["timeseries/v/" * string($iter)][1, 1, :]
fig = Figure(resolution = (1000, 1000))
title_u = @lift(@sprintf("Zonal velocity at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
title_v = @lift(@sprintf("Meridional velocity at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
ax_u = Axis(fig[1,1], xlabel = "u [m/s]", ylabel = "z [m]", title=title_u)
ax_v = Axis(fig[2,1], xlabel = "v [m/s]", ylabel = "z [m]", title=title_v)
numerical_u = lines!(ax_u, u, z,  linewidth=4, label="u")
numerical_v = lines!(ax_v, v, z,  linewidth=4, label="v")
Ekman_u     = lines!(ax_u, u_Ekplt, z,  linewidth=4, label="u linear")
Ekman_v     = lines!(ax_v, v_Ekplt, z,  linewidth=4, label="v linear")
axislegend(ax_u)
axislegend(ax_v)
#xlims!(ax_u, -0.15, 0.15)
#xlims!(ax_v, -0.15, 0.15)
display(fig)

output_prefix = ""
record(fig, "surface_Ekman_layer_output.mp4", iters[2:end], framerate=6) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end
