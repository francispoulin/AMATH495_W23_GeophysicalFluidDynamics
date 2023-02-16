using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode
using SeawaterPolynomials
using CairoMakie, JLD2, Printf

Nx, Nz = 100, 200         
Lx, Lz = 40kilometers, 1kilometers  
Δt, stop_time = 30, 4days

coriolis = FPlane(rotation_rate= 7.2921E-5, latitude=-60)

eos = SeawaterPolynomials.TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(equation_of_state=eos)

grid = RectilinearGrid(CPU();
                       size = (Nx, Nz), 
                          x = (0, Lx),
                          z = (-Lz, 0),
                   topology = (Periodic, Flat, Bounded))
                   
parameters = (cᴰ = 2.5e-3,
              ρₐ = 1.225,
              ρ₀ = 1030.0,        
              f  = coriolis.f,
              C  = 1.0,
              W  = 100,
              Cw = 3930,
              Δt = Δt)

### Surface forcing of temperature
dTdz = 0.0025 
dSdz = -0.001
@inline function T_flux(i, j, grid, clock, model_fields, p) 
        x   = xnode(Center(), i, grid)    
        t   = clock.time
        Lx  = grid.Lx
        dTH = p.W * p.Δt/(p.ρ₀ * p.Cw * grid.Δzᵃᵃᶜ)
        axx = exp.(-(8*x/Lx .- 8/2).^2)
        att = (p.C - cos(2π*t/1days)) 
        return dTH * axx * att    
    end
T_flux_bc = FluxBoundaryCondition(T_flux, discrete_form=true, parameters=parameters)
T_bcs = FieldBoundaryConditions(top = T_flux_bc,
                             bottom = GradientBoundaryCondition(dTdz))

model = NonhydrostaticModel(; grid, buoyancy,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S),
                            coriolis = coriolis,
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (T=T_bcs,)
)

Tᵢ(x, y, z) = 0  + dTdz * z 
Sᵢ(x, y, z) = 34 + dSdz * z 

set!(model, T=Tᵢ, S=Sᵢ)
simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|T|) = %.1e C, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(sim.model.tracers.T), 
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(1hours))
simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers), 
                                                      schedule = TimeInterval(0.2hour),
                                                       filename="surface_cooling.jld2", 
                                            overwrite_existing = true)

run!(simulation)

filename = "surface_cooling.jld2"
file=jldopen(filename)

xw, yw, zw = nodes((Center, Center, Face), grid)
xT, yT, zT = nodes((Center, Center, Center), grid)

iter = Observable(0)
iters = parse.(Int, keys(file["timeseries/t"]))
w_field = @lift file["timeseries/w/" * string($iter)][:, 1, :]
T_field = @lift file["timeseries/T/" * string($iter)][:, 1, :]
S_field = @lift file["timeseries/S/" * string($iter)][:, 1, :]
fig = Figure(resolution = (2000, 1000))
title_w = @lift(@sprintf("w at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
title_T = @lift(@sprintf("T at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
title_S = @lift(@sprintf("S at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
ax_w = Axis(fig[1,1], xlabel = "x (km)", ylabel = "z (km)", title=title_w)
ax_T = Axis(fig[2,1], xlabel = "x (km)", ylabel = "z (km)", title=title_T)
ax_S = Axis(fig[3,1], xlabel = "x (km)", ylabel = "z (km)", title=title_S)
heatmap_w = heatmap!(ax_w, xw/kilometers, zw/kilometers, w_field, colorrange=(-0.003, 0.003), colormap=:balance)
heatmap_T = heatmap!(ax_T, xT/kilometers, zT/kilometers, T_field, colorrange=(-2, 0), colormap=:balance)
heatmap_S = heatmap!(ax_S, xT/kilometers, zT/kilometers, S_field, colorrange=(34, 35), colormap=:balance)
Colorbar(fig[1,2], heatmap_w, label="w", width=25)
Colorbar(fig[2,2], heatmap_T, label="T", width=25)
Colorbar(fig[3,2], heatmap_S, label="S", width=25)
display(fig)

output_prefix = "surface_forcing"
record(fig, output_prefix * ".mp4", iters[2:end], framerate=6) do i
    @info "Plotting iteration $i of $(iters[end])..."
    iter[] = i
end

close(file)
