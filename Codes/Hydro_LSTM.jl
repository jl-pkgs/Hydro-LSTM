using Flux
using LinearAlgebra
using Functors

struct HydroLSTMCell
    weight_input
    weight_recur
    bias
    state_size::Int
    input_size::Int
end

# 显式指定只有这三个权重矩阵是可训练的
Functors.@functor HydroLSTMCell (weight_input, weight_recur, bias)

function HydroLSTMCell(input_size::Int, state_size::Int)
    if state_size == 1
        weight_input = Flux.glorot_uniform(4, input_size)
    else
        weight_input = Flux.glorot_uniform(state_size, 4, input_size)
    end
    weight_recur = Flux.glorot_uniform(4, state_size)
    bias = Flux.glorot_uniform(4, state_size)
    return HydroLSTMCell(weight_input, weight_recur, bias, state_size, input_size)
end

function step_cell(m::HydroLSTMCell, x_t, curr_h, curr_c)
    if m.state_size == 1
        gates = m.weight_recur * curr_h .+ m.bias .+ m.weight_input * x_t
    else
        # 向量化版本的循环，避免原地修改（mutation）以提高速度
        # weight_input: (状态大小, 4, 输入大小)
        # x_t: (输入大小,)
        wi_reshaped = reshape(m.weight_input, m.state_size * 4, m.input_size)
        sum1_flat = wi_reshaped * x_t
        sum1 = reshape(sum1_flat, 4, m.state_size)
        
        h_diag = diagm(vec(curr_h))
        gates = m.weight_recur * h_diag .+ m.bias .+ sum1
    end
    
    f = sigmoid.(gates[1:1, :])
    i = sigmoid.(gates[2:2, :])
    o = sigmoid.(gates[3:3, :])
    g = tanh.(gates[4:4, :])
    
    if m.state_size == 1
        next_c = f .* curr_c .+ i .* g
    else
        c_diag = diagm(vec(curr_c))
        next_c = (f * c_diag) .+ (i .* g)
    end
    
    next_h = o .* tanh.(next_c)
    return next_h', next_c'
end

function (m::HydroLSTMCell)(x, h_0, c_0)
    # x: (输入大小, 批次大小)
    batch_size = size(x, 2)
    
    # 使用循环收集状态。为了避免 Zygote 不支持的 push! 原地修改，
    # 我们可以使用函数式方法或直接返回最终序列。
    # 因为回归层需要所有的 h_t：
    states = []
    curr_h, curr_c = h_0, c_0
    
    # 我们使用一个小技巧：先收集到列表中，然后使用 vcat 合并。
    # Zygote 有时处理这种方式比处理预分配数组更好，
    # 但实际上最好的方法是使用 Zygote 支持的递归扫描或循环。
    
    # 尝试使用带有函数式状态更新的简单循环
    all_h = [h_0]
    for i in 1:batch_size
        curr_h, curr_c = step_cell(m, x[:, i], curr_h, curr_c)
        all_h = [all_h..., curr_h]
    end
    return reduce(hcat, all_h), curr_h, curr_c
end

mutable struct ModelHydroLSTM
    hydro_lstm::HydroLSTMCell
    regression
    state_size::Int
    h_t
    c_t
end

# 显式指定可训练参数
Functors.@functor ModelHydroLSTM (hydro_lstm, regression)

function ModelHydroLSTM(input_size::Int, state_size::Int)
    cell = HydroLSTMCell(input_size, state_size)
    reg = Dense(state_size, 1)
    return ModelHydroLSTM(cell, reg, state_size, nothing, nothing)
end

function (m::ModelHydroLSTM)(x, epoch)
    if epoch == 1 || isnothing(m.h_t)
        h_0 = zeros(eltype(x), m.state_size, 1)
        c_0 = zeros(eltype(x), m.state_size, 1)
    else
        h_0 = m.h_t
        c_0 = m.c_t
    end
    
    h_seq, last_h, last_c = m.hydro_lstm(x', h_0, c_0)
    
    # 更新状态（在梯度计算之外或由 Flux 处理）
    # 使用 Array() 获取原始数据以断开梯度链，实现截断反向传播 (TBPTT)
    # 这与 Python 中的 self.h_t.data[-1] 行为一致
    m.h_t = Array(last_h)
    m.c_t = Array(last_c)
    
    q_t = m.regression(h_seq)'
    return q_t[2:end, :]
end
