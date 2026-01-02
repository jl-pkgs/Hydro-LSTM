using NPZ, Statistics, Flux, MLUtils, Plots, LinearAlgebra, Printf, UnPack, Random
import HydroTools: of_NSE

# 设置随机种子以确保结果可复现
Random.seed!(1234)

include("../Codes/Hydro_LSTM.jl")

# --- 配置参数 ---
config = (
  code="11523200",
  cells=1,
  memory=64,
  epochs=50,
  lr=1e-3,
  batch_size=8
)
@unpack code, cells, memory, epochs, lr, batch_size = config

# --- 数据加载 ---
println("Loading preprocessed data...")
data = Dict(Symbol(k) => v for (k, v) in npzread("data/preprocessed_data.npz"))
@unpack X_train, Y_train, X_valid, Y_valid, y_stats = data
y_max, y_min, y_mean = y_stats

# 数据加载器 (Flux 期望特征在第一维)
shuffle = false
train_loader = DataLoader((collect(X_train'), collect(Y_train')); batchsize=batch_size, shuffle)
X_v, Y_obs_v = collect(X_valid'), collect(Y_valid')

# 初始化模型
model = ModelHydroLSTM(2 * (memory + 1), cells)
opt = Flux.setup(Flux.Adam(lr), model)

# --- 损失函数 ---
function loss_fn(y_obs, y_sim)
  Flux.huber_loss(y_sim, y_obs) # 默认 Huber 损失
  # -of_NSE(y_obs[:], y_sim[:]) # NSE 损失
end

# 训练循环
println("Starting training...")
for epoch in 1:epochs
  loss = 0.0
  for (x, y_obs) in train_loader
    # 每个 batch 重置状态，因为 shuffle=true
    if shuffle
      model.h_t = nothing
      model.c_t = nothing
    end

    l, grads = Flux.withgradient(m -> loss_fn(y_obs', m(x', 1)), model)
    Flux.update!(opt, model, grads[1])
    loss += l
  end

  if epoch == 1 || epoch % 5 == 0
    v_loss = loss_fn(Y_obs_v', model(X_v', 0))
    @printf("Epoch %3d | Train Loss: %.6f | Valid Loss: %.6f\n", epoch, loss / length(train_loader), v_loss)
  end
end

# 模型评估
y_sim_n = model(X_v', 0)
y_sim = y_sim_n .* (y_max - y_min) .+ y_mean
y_obs = Y_obs_v' .* (y_max - y_min) .+ y_mean

rmse = sqrt(mean((y_sim .- y_obs) .^ 2))
mae = mean(abs.(y_sim .- y_obs))
r2 = 1 - sum((y_obs .- y_sim) .^ 2) / sum((y_obs .- mean(y_obs)) .^ 2)
@printf("\nResults: RMSE=%.4f, MAE=%.4f, R2=%.4f\n", rmse, mae, r2)

# 绘图
p = plot(y_obs[1:200], label="Observed", alpha=0.7)
plot!(p, y_sim[1:200], label="Simulated", alpha=0.7)
title!("Catchment $code - Julia HydroLSTM")
savefig("example01_julia_results.png")
println("Plot saved as 'example01_julia_results.png'")
