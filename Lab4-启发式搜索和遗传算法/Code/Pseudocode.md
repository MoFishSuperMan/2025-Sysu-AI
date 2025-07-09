```python
Algorithm 1: A_star
Input:  s_0
Return: 
Initialize: frontier ← [s_0]    // open表，存储待探索节点
            close ← None        // close表，储存已探索节点
while frontier not empty do
    // 从开放表选取f值最小的节点
    s ← frontier.pop() with the lowest f(s)
    // 将s加入close表
    close ← s
    // 遍历s的所有下一个状态
    for s' in next_states do
        // 如果s'不在close表里，把s'加入到frontier表中
        if s' not in close then
            frontier ← s'
        end if
        // 如果s'是目标，那么返回搜索路径
        if s' == target then
            return search_path
        end if
    end for
end while
return None

Algorithm 2: IDA_star
Input:  s_0
Return: 
Initialize: bound ← Heurisic(s_0) + g(s_0) // 初始bound值
while True do
    initialize: close ← None
    // 调用dfs获得最小花费
    min_cost ← dfs(s_0)
    // 如果最小花费为-1 表示已经找到目标，返回搜索路径
    if min_cost == -1 then
        return search_path
    // 更新bound值为上一次迭代的最小花费
    bound ← min_cost
end while
return None

function dfs
Input:  s,bound
Return: min_cost
// 递归结束条件：如果当前状态f值大于上限bound返回f
if f(s) > bound then
    return f(s)
end if
// 递归结束条件：如果找到目标，返回-1 表示结束
if s == target then
    return -1
end if
// 遍历s的所有下一个状态
for s' in next_states do
    if s' not in close then
        close ← s'
        // 递归调用dfs
        cost ← dfs(s')
        if cost == -1 then
            return -1
        end if
        // 返回cost和min_cost的最小值
        if cost < min_cost then
            min_cost ← cost
        end if
    end if
end for
return min_cost


Algorithm 3: TSP of GeneticAlgorithm
Input:  Coordinates of n cities
Return: best solution
Initialize: population_size // 种群大小
            num_iterations  // 迭代次数
            mutation_rate   // 变异率
            P(0) ← encoding routinue
    for t=0 to num_interations do
        for i=0 to population_size do
            // 依据fitness，从P(t)中选择两个个体作为亲本
            parent1,parent2 ← selction(P(t), 2)
            // 两个亲本crossover产生两个子代
            child1,child2 ← crossover(parent1,parent2)
            // 两个子代以一定的变异率产生变异
            child1,child2 ← mutation(child1,child2,mutation_rate)
            // 产生的所有子代构成C(t)
            C(t) ← child1,child2
        end for
    // 根据fitness，从C(t)∪P(t)中选取population_size个个体构成下一代
    P(t+1) ← selction(C(t)∪P(t), population_size)
    end for
    return best soulutin in last generation
```