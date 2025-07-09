import heapq
import time
import tracemalloc
def oneD2twoD(index):
    x=index//4
    y=index%4
    return x,y

def misplaced(puzzle):
    '''启发式函数1：不在正确位置的方块个数'''
    value=0
    for index,num in enumerate(puzzle):
        if num == 0:
            continue
        target_x,target_y=target_coord[num]
        current_x,current_y=oneD2twoD(index)
        if (target_x,target_y) != (current_x,current_y):
            value+=1
    return value

def manhattan(puzzle):
    '''启发式函数2：曼哈顿距离'''
    value=0
    for index,num in enumerate(puzzle):
        if num == 0:
            continue
        target_x,target_y=target_coord[num]
        current_x,current_y=oneD2twoD(index)
        value+=abs(target_x-current_x)+abs(target_y-current_y)
    return value

def optimized_manhattan(next_puzzle,move_block,pre_value):
    '''启发式函数3：记忆性曼哈顿距离'''
    cur_value=0
    pre_index=next_puzzle.index(0)
    cur_index=next_puzzle.index(move_block)
    # 只需要计算移动的两个位置的曼哈顿距离的变化值即可
    target_x,target_y=target_coord[move_block]
    pre_x,pre_y=oneD2twoD(pre_index)
    cur_x,cur_y=oneD2twoD(cur_index)
    pre_h=abs(pre_x-target_x)+abs(pre_y-target_y)
    cur_h=abs(cur_x-target_x)+abs(cur_y-target_y)
    # 借用上一个状态的h值加上交换两个数字的h值得变化量得到新的h值
    cur_value=pre_value-pre_h+cur_h
    return cur_value

def linear_conflict_manhattan(puzzle):
    '''启发式函数4：线性冲突优化的曼哈顿距离'''
    value=0
    for index,num in enumerate(puzzle):
        if num == 0:
            continue
        target_x,target_y=target_coord[num]
        current_x,current_y=oneD2twoD(index)
        '''
        对于同一行或者同一列的两个数字，如果他们分别出现在了对方的移动路径中，
        发生线性冲突，也就是至少需要多移动两次才能复原，故给h值加上惩罚项2
        '''
        if target_x == current_x:
            for k in range(current_y+1,4):
                other_x,other_y=target_coord[puzzle[current_x*4+k]]
                if other_y < target_y and other_x == current_x and puzzle[current_x*4+k]!=0:
                    value+=2
        if target_y == current_y:
            for k in range(current_x+1,4):
                other_x,other_y=target_coord[puzzle[k*4+current_y]]
                if other_x < target_x and other_y == current_y and puzzle[k*4+current_y]!=0:
                    value+=2
        value+=abs(target_x-current_x)+abs(target_y-current_y)
    return value

def generate_path(puzzle,parents):
    '''实现从目标回溯路径的函数'''
    path=[]
    while parents[puzzle] != (None,None):
        pre_puzzle,move_block=parents[puzzle]
        path.append(move_block)
        puzzle=pre_puzzle
    return path[::-1]

def get_next_states(puzzle):
    '''获得当前状态的邻接状态'''
    next_states=[]
    current_zero_index=puzzle.index(0)
    current_x,current_y=oneD2twoD(current_zero_index)
    # x,y坐标的变化量
    for dx,dy in [(0,1),(0,-1),(-1,0),(1,0)]:
        # 新坐标等于原坐标加上变化量
        next_x,next_y=current_x+dx,current_y+dy
        # 如果新坐标合法
        if 0 <= next_x < 4 and 0 <= next_y <4:
            new_puzzle=list(puzzle)
            next_zero_index=next_x*4+next_y
            # 记录移动的数字
            move_block=puzzle[next_zero_index]
            # 交换新坐标和原坐标的数字
            new_puzzle[current_zero_index],new_puzzle[next_zero_index]=new_puzzle[next_zero_index],new_puzzle[current_zero_index]
            next_states.append((tuple(new_puzzle),move_block))
    return next_states

def A_star(puzzle,target):
    # 初始化初始状态的数据结构、g、h值
    puzzle=tuple([puzzle[i][j] for i in range(4) for j in range(4)])
    target=tuple([target[i][j] for i in range(4) for j in range(4)])
    init_g=0
    # 启发式函数，这里选用线性冲突优化的曼哈顿距离
    init_h=linear_conflict_manhattan(puzzle)
    # frontier表
    frontier=[]
    # close表
    visited={puzzle:0}
    # 父亲节点列表
    parents=dict()
    parents[puzzle]=(None,None)
    # 初始化堆，元素为(f,puzzle,g,h)
    heapq.heappush(frontier,(init_g+init_h,puzzle,init_g,init_h))
    # A_star算法核心循环
    while frontier:
        # 当前节点
        _,cur_puzzle,g,h=heapq.heappop(frontier)
        # 找到目标返回路径
        if cur_puzzle == target:
            return generate_path(target,parents)
        # 遍历所有邻接节点
        for next_puzzle,move_block in get_next_states(cur_puzzle):
            # 如果不在close表，或者出现在close但是g值更优，加入frontier表
            if next_puzzle not in visited or visited[next_puzzle]>g+1:
                parents[next_puzzle]=(cur_puzzle,move_block)
                visited[next_puzzle]=g+1
                #next_h=optimized_manhattan(next_puzzle,move_block,h)
                next_h=linear_conflict_manhattan(next_puzzle)
                heapq.heappush(frontier,(g+1+next_h,next_puzzle,g+1,next_h))
    return []

def dfs(puzzle,target,g,visited,parents,bound):
    # 计算当前状态的f值，这里选用线性冲突优化的曼哈顿距离
    f=linear_conflict_manhattan(puzzle)+g
    # 如果f值超过了当前的界限，返回f值
    if f>bound:
        return f
    # 如果当前状态等于目标状态，返回-1表示找到目标
    if puzzle == target:
        return -1
    min_cost=float('inf')
    # 遍历当前状态的所有可能的下一个状态
    for next_puzzle,move_block in get_next_states(puzzle):
        # 如果不在close表，或者出现在close但是g值更优，加入frontier表
        if next_puzzle not in visited or visited[next_puzzle] > g+1:
            visited[next_puzzle]=g+1
            parents[next_puzzle]=(puzzle,move_block)
            # 递归调用dfs，优先扩展出现的节点
            cost=dfs(next_puzzle,target,g+1,visited,parents,bound)
            if cost == -1:
                return -1
            # 如果当前的代价小于最小代价，更新最小代价
            if cost < min_cost:
                min_cost=cost
    return min_cost

def IDA_star(puzzle,target):
    puzzle=tuple([puzzle[i][j] for i in range(4) for j in range(4)])
    target=tuple([target[i][j] for i in range(4) for j in range(4)])
    # 初始上限
    bound=linear_conflict_manhattan(target)
    # IDA_star核心循环
    while 1:
        # 每次循环从初始点开始搜索，初始化close表
        visited={puzzle:0}
        # 记录父亲节点
        parents=dict()
        parents[puzzle]=(None,None)
        # 调用dfs深度搜索
        min_cost=dfs(puzzle,target,0,visited,parents,bound)
        # 找到目标，返回搜索路径
        if min_cost == -1:
            return generate_path(target,parents)
        # 若未找到更新上限bound迭代加深上限f值
        bound=min_cost
    return []

if __name__ == '__main__':
    puzzles=[
        [[1,2,3,4],[5,6,7,8],[9,10,11,12],[0,13,14,15]],    #3
        [[1,2,4,8],[5,7,11,10],[13,15,0,3],[14,6,9,12]],    #22
        [[14,10,6,0],[4,9,1,8],[2,3,5,11],[12,13,7,15]],    #49
        [[5,1,3,4],[2,7,8,12],[9,6,11,15],[0,13,10,14]],    #15
        [[6,10,3,15],[14,8,7,11],[5,1,0,2],[13,12,9,4]],    #48
        [[11,3,1,7],[4,6,8,2],[15,9,10,13],[14,12,5,0]],    #56
        [[0,5,15,14],[7,9,6,13],[1,2,12,10],[8,11,4,3]]     #62
        ]
    target=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]
    target_coord={target[i][j]:(i,j) for i in range(4) for j in range(4)}
    for i in range(len(puzzles)):
        start=time.perf_counter()
        result=(IDA_star(puzzles[i],target))
        end=time.perf_counter()
        tracemalloc.start()
        result=(IDA_star(puzzles[i],target))
        current,peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(result)
        print(f'步数:{len(result)}')
        print(f'Runing time:{end-start:.8f}s')
        print(f'Peak memory:{peak/1024/1024:.4f}MB')
        print()