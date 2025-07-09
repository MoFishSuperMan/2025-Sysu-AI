
from string import ascii_lowercase
from collections import deque

def numbering_formula(clause,serial):
    '''
    为输入的子句进行编号
    如果只有一个原子公式那么该原子公式的编号就是该子句的编号,
    如果一个子句有多个原子公式那么就在子句编号的基础上用小写字母a,b,c...加以区分
    
    args:
        clause: 要编号的子句
        clause_dict: 子句集的编号字典
        
    returns:
        原子式到唯一标识的映射字典
    '''
    if len(clause)>1: 
        formulas_dict={item:f'{serial}{ascii_lowercase[i]}' for i,item in enumerate(clause)}
    else:
        formulas_dict={item:serial for item in clause}
    return formulas_dict

def formula_pair(clause1,clause2):
    '''
    从两个子句中分别寻找原子公式互补对：
    包括相同的原子公式及其对应的原子公式否定
    
    args(tuple,tuple):
        clause1: 子句1
        clause2: 子句2
        
    return(str,str):
        找到的原子公式互补对
        如果不存在则返回None
    
    '''
    for clause_formula in clause1:
        if ('~'+clause_formula) in clause2:
            return clause_formula,'~'+clause_formula
        elif clause_formula.startswith('~') and clause_formula[1:] in clause2:
            return clause_formula,clause_formula[1:]
    return None,None

def resolve(clause1,clause2,formula,complt_formula):
    '''
    实现单步归结操作
    args(tuple,tuple,str,str):
        clause1: 子句1
        clause2: 子句2
        formula,complt_formula: 要消去的原子公式及其否定

    returns(tuple):
        new_clause:归结新产生的子句
    
    '''

    new_clause=tuple(sorted((set(clause1+clause2))-{formula,complt_formula}))  #利用set实现差集的运算
    return new_clause

def generate_steps(resolution_path,clause_dict,parent_info,KB):
    steps=[]
    #将子句条件加入步骤列表
    for i,clause in enumerate(KB):
        steps.append(clause)
    #从头遍历归结路径
    for clause in resolution_path:
        if clause not in list(KB):
            #获得亲本子句的信息，包括亲本子句，置换集以及消去的互补原子公式
            clause1,clause2=parent_info[clause]['parents']
            formula1,formula2=parent_info[clause]['formulas']
            #为子句的每一个项进行编号，用于后面的规格化输出
            clause1_dict=numbering_formula(clause1,clause_dict[tuple(clause1)])
            clause2_dict=numbering_formula(clause2,clause_dict[tuple(clause2)])
            #利用差集的进行归结
            new_clause=resolve(clause1,clause2,formula1,formula2)
            #为新子句进行编号
            clause_dict.update({new_clause:len(clause_dict)+1})
            #规格化步骤的格式即 步骤号R[亲本子句的编号]{应用的置换} = 归结后的子句
            new_step=f'R[{",".join(sorted(map(str,[clause1_dict[formula1],clause2_dict[formula2]])))}] = {new_clause}'
            steps.append(new_step)
    return steps

def generate_resolution_path(new_clause,parent_info):
    resolution_path=[]
    #采用BFS的方法从空子句()回溯到初始条件获得归结路径，这时归结路径是反着的
    queue=deque()
    queue.append(new_clause)
    while len(queue)>0:
        current_clause=queue.popleft()
        resolution_path.append(current_clause)
        for parent_clause in parent_info[current_clause]['parents']:
            if parent_clause is not None:
                queue.append(parent_clause)
    #将归结路径reversed后，转成字典后再转成list的方法去重，这种方法可以保证去重前后的顺序不会改变
    resolution_path=list(dict.fromkeys(reversed(resolution_path)))
    return resolution_path

def ResolutionProp(KB):
    clauses=sorted(list(KB)) #子句集合包括初始条件KB中的以及归结得到的
    parent_info={clause:{'parents':(None,None),'formulas':(None,None)} for clause in clauses}
    generations=[clauses]   #归结代数
    while 1:
        current_generation=generations[-1]
        next_generation=[]
        #取出子句集中的两个不重复的子句
        new_clause=()
        for pre_generation in generations:     #遍历过往的每一代
            for clause1 in current_generation: #clause1从当代中寻找
                for clause2 in pre_generation: #clause2从过往的代数寻找
                    if clause1 == clause2:
                        continue
                    #寻找两个子句是否存在互补的原子公式
                    formula1,formula2=formula_pair(clause1,clause2)
                    #没有原子公式那么跳过剩下的步骤
                    if not formula1:
                        continue
                    #对置换后的子句进行归结
                    new_clause=resolve(clause1,clause2,formula1,formula2)
                    #如果归结出了新的子句
                    if new_clause not in clauses:
                        clauses.append(new_clause)      #加入到子句集最后 广度优先
                        #clauses.insert(0,new_clause)   #加入到子句集最前 深度优先
                        parent_info[new_clause]={
                            'parents':(clause1,clause2),
                            'formulas':(formula1,formula2)
                        }#记录该子句的亲本信息，包括亲本子句以及互补的原子公式
                        #将新的子句加入到下一代的子句集中
                        next_generation.append(new_clause)
                        #如果新的子句是空子句说明归结完成
                        if new_clause == ():
                            #生成归结路径，过滤掉无用子句
                            resolution_path=generate_resolution_path(new_clause,parent_info)
                            clause_dict={clause:index+1 for index,clause in enumerate(KB)} #编号
                            #生成归结步骤用于输出
                            steps=generate_steps(resolution_path,clause_dict,parent_info,KB)
                            return steps
        if not next_generation:  #如果没有新的子句生成，结束循环
            break
        #将新生成的子句集合添加到generations中
        generations.append(next_generation)
    return steps

if __name__ == '__main__':
    '''
    测试程序

    测试样例:
    KB = {("FirstGrade",),("~FirstGrade","Child"),("~Child",)}
    '''
    KB = {("FirstGrade",),("~FirstGrade","Child"),("~Child",)}
    print("Input:\nKB = ",KB,"\nOutput:")
    for i,step in enumerate(ResolutionProp(KB)):
        print(i+1,step)