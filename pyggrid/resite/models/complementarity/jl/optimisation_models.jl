using JuMP
using Gurobi

function solve_MILP(D::Array{Float64, 2}, c::Float64, n::Float64, solver::String)

  W = size(D)[1]
  L = size(D)[2]

  if solver == "Gurobi"
    MILP_model = Model(optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => 7200., "MIPGap" => 0.05))
  else
      println("Please use Cbc or Gurobi")
      throw(ArgumentError)
  end

  @variable(MILP_model, x[1:L], Bin)
  @variable(MILP_model, 0 <= y[1:W] <= 1)

  @constraint(MILP_model, cardinality, sum(x) == n)
  @constraint(MILP_model, covering, D * x .>= c * y)

  @objective(MILP_model, Max, sum(y))

  optimize!(MILP_model)

  x_sol = round.(value.(x))

  return x_sol

end

function solve_MILP_partitioning(D::Array{Float64, 2}, c::Float64, n::Array{Int64, 1}, partitions_indices::Dict{Int64, Int64}, solver::String)

  W = size(D)[1]
  L = size(D)[2]
  P = length(n)

  # Computes number of locations in each partition
  cnt = zeros(Int64, P)
  for i = 1:L
    cnt[partitions_indices[i]] += 1
  end

  # Computes indices of partitions
  ind_part = Vector{Int64}(undef, P+1)
  ind_part[1] = 1
  for i = 1:P
    ind_part[i+1] = ind_part[i] + cnt[i]
  end

  # Selects solver
  if solver == "Gurobi"
    MILP_model = Model(optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => 7200., "MIPGap" => 0.01))
  else
      println("Please use Cbc or Gurobi")
      throw(ArgumentError)
  end

  # Defines variables
  @variable(MILP_model, x[1:L], Bin)
  @variable(MILP_model, 0 <= y[1:W] <= 1)

  # Defines Constraints
  @constraint(MILP_model, cardinality[i=1:P], sum(x[ind_part[i]:(ind_part[i+1]-1)]) == n[i])
  @constraint(MILP_model, covering, D * x .>= c * y)

  # Defines objective function
  @objective(MILP_model, Max, sum(y))

  # Solves model
  optimize!(MILP_model)

  # Extracts solution
  x_sol = round.(value.(x))

  return x_sol

end

function solve_IP(D::Array{Float64, 2}, c::Float64, n::Float64, solver::String, timelimit::Float64, gap::Float64)

  W = size(D)[1]
  L = size(D)[2]

  println("Building IP Model")
  if solver == Gurobi
    IP_model = Model(optimizer_with_attributes(solver.Optimizer, "TimeLimit" => timelimit, "MIPGap" => gap))
  else
      println("Please use Cbc or Gurobi")
      throw(ArgumentError)
  end

  @variable(IP_model, x[1:L], Bin)
  @variable(IP_model, y[1:W], Bin)

  @constraint(IP_model, cardinality, sum(x) == n)
  @constraint(IP_model, covering, D * x .>= c * y)

  @objective(IP_model, Max, sum(y))

  optimize!(IP_model)

  x_sol = value.(x)
  y_sol = value.(y)

  return x_sol, y_sol

end

function warm_start_IP(D::Array{Float64, 2}, c::Float64, n::Float64, x_init::Array{Float64, 1}, y_init::Array{Float64, 1}, solver::String, timelimit::Float64, gap::Float64)

W = size(D)[1]
L = size(D)[2]

println("Warmstarting IP Model")
if solver == Gurobi
  IP_model = Model(optimizer_with_attributes(solver.Optimizer, "TimeLimit" => timelimit, "MIPGap" => gap))
else
    println("Please use Cbc or Gurobi")
    throw(ArgumentError)
end

@variable(IP_model, x[1:L], Bin)
@variable(IP_model, y[1:W], Bin)

set_start_value.(x, x_init)
set_start_value.(y, y_init)

@constraint(IP_model, cardinality, sum(x) == n)
@constraint(IP_model, covering, D * x .>= c * y)

@objective(IP_model, Max, sum(y))

optimize!(IP_model)

x_sol = value.(x)
y_sol = value.(y)

return x_sol, y_sol

end