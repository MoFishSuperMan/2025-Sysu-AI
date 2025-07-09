from string import ascii_lowercase
from collections import deque
import MGU

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

def find_formula_pair(clause1,clause2):
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
    new_clause=[]
    #利用set实现差集的运算
    new_clause=tuple(sorted((set(clause1+clause2))-{formula,complt_formula}))
    return new_clause

def MGU_clause(clause1,clause2):
    substitution_ways=[{}]
    #遍历两个子句中所有的原子公式组合
    for formula1 in clause1:
        for formula2 in clause2:
            #调用MGU函数实现两个原子公式的最一般合一置换集
            #加入到置换方法列表中
            substitution_ways.append(MGU.MGU(formula1,formula2))
    return substitution_ways

def replace_clause(clause,substitution):
    if not substitution:
        return clause
    new_clause=[]
    clause=list(clause)
    #遍历子句的每个原子公式
    for formula in clause:
        #调用MGU中的replace_variable函数，将每个原子公式都进行置换
        new_clause.append(MGU.replace_variable(formula,substitution))
    return tuple(new_clause)

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
            sub=parent_info[clause]['sub']
            formula1,formula2=parent_info[clause]['formulas']
            #应用置换集获得置换后的子句
            new_clause1=replace_clause(clause1,sub)
            new_clause2=replace_clause(clause2,sub)
            #为子句的每一个项进行编号，用于后面的规格化输出
            clause1_dict=numbering_formula(new_clause1,clause_dict[tuple(clause1)])
            clause2_dict=numbering_formula(new_clause2,clause_dict[tuple(clause2)])
            #为新子句进行编号
            clause_dict.update({clause:len(clause_dict)+1})
            #规格化步骤的格式即 步骤号R[亲本子句的编号]{应用的置换} = 归结后的子句
            if not sub:
                new_step=f'R[{",".join(sorted(map(str,[clause1_dict[formula1],clause2_dict[formula2]])))}] = {clause}'
            else:
                substitution_str='{'+','.join([f'{old}={new}' for old, new in sub.items()])+'}'
                new_step=f'R[{",".join(sorted(map(str,[clause1_dict[formula1],clause2_dict[formula2]])))}]{substitution_str} = {clause}'
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

def FirstOrderLogicResolution(KB):
    clauses=sorted(list(KB)) #子句集合包括初始条件KB中的以及归结得到的
    parent_info={clause:{'parents':(None,None),'sub':None,'formulas':(None,None)} for clause in clauses}
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
                    #获得两个子句的所有可能的置换的方法，一次归结只能够合一两个子句的两个原子公式
                    substitution_ways=MGU_clause(clause1,clause2)
                    #遍历所有的可能的两个子句的置换
                    for sub in substitution_ways:
                        #应用置换集获得置换后的子句
                        new_clause1=replace_clause(clause1,sub)
                        new_clause2=replace_clause(clause2,sub)
                        #寻找两个子句是否存在互补的原子公式
                        formula1,formula2=find_formula_pair(new_clause1,new_clause2)
                        #没有原子公式那么跳过剩下的步骤
                        if not formula1:
                            continue
                        #对置换后的子句进行归结
                        new_clause=resolve(new_clause1,new_clause2,formula1,formula2)
                        #如果归结出了新的子句
                        if new_clause not in clauses:
                            clauses.append(new_clause)      #加入到子句集最后 广度优先
                            #clauses.insert(0,new_clause)   #加入到子句集最前 深度优先
                            parent_info[new_clause]={
                                'parents':(clause1,clause2),
                                'sub':sub,
                                'formulas':(formula1,formula2)
                            }#记录该子句的亲本信息，包括亲本子句，置换以及互补的原子公式
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
    '''
    #KB1={("FirstGrade",),("~FirstGrade","Child"),("~Child",)}
    KB1={('GradStudent(sue)',),('~GradStudent(x)','Student(x)'),
        ('~Student(x)','HardWorker(x)'),('~HardWorker(sue)',)}
    KB2={('A(tony)',),('A(mike)',),('A(john)',),('L(tony,rain)',),
        ('L(tony,snow)',),('~A(x)','S(x)','C(x)'),('~C(y)','~L(y,rain)'),
        ('L(z,snow)','~S(z)'),('~L(tony,u)','~L(mike,u)'),
        ('L(tony,v)','L(mike,v)'),('~A(w)','~C(w)','S(w)')}
    KB3={('On(tony,mike)',),('On(mike,john)',),('Green(tony)',),
        ('~Green(john)',),('~On(xx,yy)','~Green(xx)','Green(yy)')}
    KBs = [KB1,KB2,KB3]
    for i in range(len(KBs)):
        print(f'KB{i+1}:')
        print("Input:\nKB = ",KBs[i],"\nOutput:")
        for i,step in enumerate(FirstOrderLogicResolution(KBs[i])):
            print(i+1,step)
        print("----------------")
