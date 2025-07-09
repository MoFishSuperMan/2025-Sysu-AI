import FirstOrderLogicResolution
import itertools
def linear_input(clauses,n):
    pair=[]
    original_clauses=clauses[0:n]
    for clause1 in original_clauses:
        for clause2 in (clauses):
            pair.append((clause1,clause2))
    return pair

def support_set_strategy(clauses,n):
    pair=[]
    support_set=(clauses[n-1:])
    for clause1 in (support_set):
        for clause2 in (clauses):
            pair.append((clause1,clause2))
    return pair

def bfs(clauses,n):
    pair=[]
    for clause1 in clauses:
        for clause2 in clauses:
            pair.append((clause1,clause2))
    return pair

def resolution(KB,function):
    depth=0
    clauses=(list(KB)) #子句集合包括初始条件KB中的以及归结得到的
    while 1:
        new_clause=()
        for clause1,clause2 in function(clauses,len(KB)):
            substitution_ways=FirstOrderLogicResolution.MGU_clause(clause1,clause2)
            for new_subititution in substitution_ways:
                new_clause1=FirstOrderLogicResolution.replace_clause(clause1,new_subititution)
                new_clause2=FirstOrderLogicResolution.replace_clause(clause2,new_subititution)
                formula1,formula2=FirstOrderLogicResolution.formula_pair(new_clause1,new_clause2)
                if not formula1:
                    continue
                new_clause=FirstOrderLogicResolution.resolve(new_clause1,new_clause2,formula1,formula2)
                if new_clause not in clauses:
                    clauses.append(new_clause)   #加入到子句集最后 广度优先
        depth+=1
        if () in clauses:
            return clauses,depth
        if not new_clause:
            break
    return clauses,depth


def dfs(KB):
    clauses=(list(KB)) #子句集合包括初始条件KB中的以及归结得到的
    depth=0
    while 1:
        new_clause=()
        #for (clause1,clause2) in linear_input(clauses,len(KB)):
        for clause1,clause2 in itertools.combinations(clauses,2):
            substitution_ways=FirstOrderLogicResolution.MGU_clause(clause1,clause2)
            for new_subititution in substitution_ways:
                break_label=False
                new_clause1=FirstOrderLogicResolution.replace_clause(clause1,new_subititution)
                new_clause2=FirstOrderLogicResolution.replace_clause(clause2,new_subititution)
                formula1,formula2=FirstOrderLogicResolution.formula_pair(new_clause1,new_clause2)
                if not formula1:
                    continue
                new_clause=FirstOrderLogicResolution.resolve(new_clause1,new_clause2,formula1,formula2)
                if new_clause not in clauses:
                    clauses.insert(0,new_clause)   #加入到子句集最后 广度优先
                    #clauses.append(new_clause)
                    if new_clause == ():
                        return clauses,depth
                    break_label=True
                    depth+=1
                    break
            if break_label:
                break
        if not new_clause:
            break
    return clauses,depth

def test(KB,n):
    KB=list(KB)
    sum1=0
    sum2=0
    sum3=0
    sum4=0
    clause1,depth1=resolution(KB,bfs)
    clause2,depth2=resolution(KB,linear_input)
    clause3,depth3=resolution(KB,support_set_strategy)
    clause4,depth4=dfs(KB)
    sum1+=(len(clause1))
    sum2+=(len(clause2))
    sum3+=(len(clause3))
    sum4+=(len(clause4))
    sums=[sum1,sum2,sum3,sum4]
    depths=[depth1,depth2,depth3,depth4]
    return sums,depths


if __name__ == '__main__':
    #KB = [('A(tony)',),('A(mike)',),('A(john)',),('L(tony,rain)',),('L(tony,snow)',),('~A(x)','S(x)','C(x)'),('~C(y)','~L(y,rain)'),('L(z,snow)','~S(z)'),('~L(tony,u)','~L(mike,u)'),('L(tony,v)','L(mike,v)'),('~A(w)','~C(w)','S(w)')]
    KB=[('~I(x)','R(x)'),('I(a)',),('~R(y)','~L(y)'),('L(a)',)]
    #KB=[('Green(tony)',),('On(mike,john)',),('On(tony,mike)',),('~Green(john)',),('~On(xx,yy)','~Green(xx)','Green(yy)')]
    functions=['bfs','linear_input','support_set_strategy','dfs']
    n=1
    sums,depth=test(KB,n)
    for i in range(4):
        print(functions[i]+':\n','产生的子句数量:',sums[i]/n,'\n','获得空子句的深度:',depth[i]/n)