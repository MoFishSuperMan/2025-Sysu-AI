```python
Algorithm: DQN
Inputs: ReplayBuffer, batch_size, target_net_update_interval
Return: policy
initialize: ReplayBuffer,
            Qnet
for ep=1,2,...,E do
    initialize env and get initial state
    for t=1,2,...,T do 
        action ← Qnet by ε-greedy policy
        next_state, reward ← by action
        ReplayBuffer.append((state, action, next_state, reward))
        sample batch_size in ReplayBuffer
        update Qnet
        if t % d ==0 then
            update target_net
    end for
end for
```