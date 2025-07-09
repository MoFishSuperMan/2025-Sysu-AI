'''
测试样例：
1.MGU('P(XX,a)','P(b,yy)')
2.MGU('P(a,xx,f(g(yy)))','P(zz,f(zz),f(uu))')

本程序所定义的函数：
extrct
is_variable
difference
'''
import re

def extract(formula):
    '''
    从原子公式中提取谓词和参数项
    如果公式是单个变量，则返回空字符串和该变量
    args(str):
    formula:原子公式字符串 
    return(str,list):
    predicate,terms:谓词,参数项列表 
    '''
    left=formula.find('(') #from left to right find 
    right=formula.rfind(')') #from right to left find
    if left==-1 or right==-1:
        return '',formula #单个变量无谓词返回空谓词以及自己本身就是自己唯一的项
    #如果原子公式以~开头，那么谓词就是从第二个字母开始到左括号的子串
    if formula.startswith('~'):
        predicate=formula[1:left]
    #如果是正文字，那么谓词就是从第一个字母到左括号的字串
    else:
        predicate=formula[0:left]
    #原子公式的项就是左括号到右括号的字串同时用逗号分割成的列表
    terms=[term for term in formula[left+1:right].split(',')]
    return predicate,terms


def is_individual(formula):
    '''
    判断一个项是否为变量
    如果项没有谓词,则认为是变量
    args(str):
    formula:项term
    return(bool):
    True:是变量
    False:不是变量
    '''
    predicate,terms=extract(formula)
    if not predicate:
        return True
    else:
        return False


def difference(formula1,formula2):
    '''
    找出两个原子公式之间的差异集D
    args(str,str):
    formula1:原子公式1
    formula2:原子公式2
    return(list):
    differences:差异集
    '''
    #提取原子公式的terms
    predicate1,terms1=extract(formula1)
    predicate2,terms2=extract(formula2)
    #如果两个公式都是变量或者常量
    if is_individual(formula1) and is_individual(formula2):
        #如果两个原子公式相等，那么递归结束返回
        if formula1==formula2 or (not re.fullmatch(r'^([d-z])(\1)?$',formula1) and not re.fullmatch(r'^([d-z])(\1)?$',formula2)) or (re.fullmatch(r'^([d-z])(\1)?$', formula1) and re.fullmatch(r'^([d-z])(\1)?$', formula2)):
            return
        else:
        #否则就说明这就是两个公式差异的地方，返回差异集
            return [formula1,formula2]
    #如果谓词不相等，那么就是差异的地方
    if predicate1!=predicate2:
        return [formula1,formula2]
    elif predicate1 == predicate2:
        differences=[]
        #如果谓词相等，那么递归的调用函数比较两个原子公式的项之间的差异
        for i in range(len(terms1)):
            if difference(terms1[i],terms2[i]):
                differences.append(difference(terms1[i],terms2[i]))
    if len(differences)>0:
        return differences[0]
    else:
        return []

def replace_variable(formula,substitution):
    '''
    对给定的公式应用置换集
    args(str,dict):
    formula:需要进行置换的公式 
    substitution:置换集 
    return(str):
    new_formula:置换后的公式
    '''
    new_formula=formula
    #遍历置换集字典的每一个items
    for old,new in substitution.items():
        #使用正则表达式将原子公式中所有的old都替换成new
        #正则表达式的好处就是不会错误的替换原子公式中的字串
        #比如要将w替换成a，那么他不会将jown中的w替换成a
        pattern = r'\b' + re.escape(old) + r'\b'
        new_formula=re.sub(pattern,new,new_formula)
    return new_formula

def composition(set1,set2): 
    '''
    实现两个置换集的复合
    args(list,list):
    set1:置换集1
    set2:置换集2
    return(list):
    set:复合后的置换集
    '''
    set={}
    #两两作用的到原始的替换集
    for old,new in set1.items():
        set.update({old:replace_variable(new,set2)})
    #删除掉有歧义的替换，即替换是一一对应的
    for old,new in set2.items():
        if old not in set:
            set.update({old:new})
    #删掉old=new的替换
    return {old:new for old,new in set.items() if old!=new}


def MGU(formula1,formula2):
    '''
    MGU算法 实现两个谓词相同的原子公式的最一般合一
    args(str,str):
    formula1:原子公式1
    formula2:原子公式2
    return(dict):
    substitution:置换集 dict{a:b}
    '''
    D=[]            #差异集D
    substitution={} #替换集σ
    while 1:
        D=difference(formula1,formula2)     #计算出两个项的的差异的地方
        if D==[]:
            break
        if is_individual(D[0]) and is_individual(D[1]):
            old,new=D[0],D[1]
            if re.fullmatch(r'^([d-z])(\1)?$', D[0]) and not re.fullmatch(r'^([d-z])(\1)?$', D[1]):
                old,new=D[0],D[1]
            elif not re.fullmatch(r'^([d-z])(\1)?$', D[0]) and re.fullmatch(r'^([d-z])(\1)?$', D[1]):
                old,new=D[1],D[0]
            elif re.fullmatch(r'^([d-z])(\1)?$', D[0]) and re.fullmatch(r'^([d-z])(\1)?$', D[1]):
                continue
            elif not re.fullmatch(r'^([d-z])(\1)?$', D[0]) and not re.fullmatch(r'^([d-z])(\1)?$', D[1]):
                continue
            substitution=composition(substitution,{old:new})
            formula1=formula1.replace(old,new)
            formula2=formula2.replace(old,new)
        elif is_individual(D[0]) and not is_individual(D[1]):
            predicate,terms=extract(D[1])
            if D[0] not in terms:
                substitution=composition(substitution,{D[0]:D[1]})
                formula1=formula1.replace(D[0],D[1])
                formula2=formula2.replace(D[0],D[1])
            else:
                return {}
        elif is_individual(D[1]) and not is_individual(D[0]):
            predicate,terms=extract(D[0])
            if D[1] not in terms:
                substitution=composition(substitution,{D[1]:D[0]})
                formula1=formula1.replace(D[1],D[0])
                formula2=formula2.replace(D[1],D[0])
            else:
                return {}
            
        else:
            break
    return substitution

if __name__ == '__main__':
    '''
    测试程序
    '''
    print("Input:",'P(xx,a)','P(b,yy)',"\nOutput:")
    print(MGU('P(xx,a)','P(b,yy)'),'\n')

    print("Input:",'P(a,xx,f(g(yy)))','P(zz,f(zz),f(uu))',"\nOutput:")
    print(MGU('P(a,xx,f(g(yy)))','P(zz,f(zz),f(uu))'))
'''
    print("Input:",'P(xx,a)','P(b,yy)',"\nOutput:")
    print(MGU('P(x,x)','P(y,f(y))'),'\n')

    print("Input:",'P(a,x,h(g(z)))','P(z,h(y),h(y))',"\nOutput:")
    print(MGU('P(a,x,h(g(z)))','P(z,h(y),h(y))'),'\n')
    
    print("Input:",'A(xx)','~A(yy)',"\nOutput:")
    print(MGU('A(xx)','~A(yy)'),'\n')
'''