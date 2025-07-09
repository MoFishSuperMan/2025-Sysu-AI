# **Algorithm 1: Propositional Logic Resolution**

```python
Algorithm 1: Propositional Logic Resolution
function ResolutionProp(KB)
    inputs: KB      //子句集
    returns: steps  //归结步骤列表
1.   var clause_dict:dict
     var clauses,steps:list
     var generations//记录归结代数
     var parent_info:dict  //记录子句亲本信息
2.   while true do
3.       for pre_generation in generations do
4.           for clause1 in current_generation do
5.               for clause2 in pre_generation do
                     //当前代一次与之前的所有代一一配对归结
6.                   formula1,formula2 ← find_formula_pair
7.                   if formula1,formula2 then  //如果找到互补对
8.                       new_clause ← resolve(new_clause1,new_clause2)//归结
9.                       if new_clause not in clausese then  //如果是新子句
                             //更新新子句和归结步骤
10.                          update(clauses,steps,parent_info)  
11.                          if new_clause == () then   //如果是空子句
                                 //生成归结路径
12.                              resolution_path ← generate_resolution_path
13.                              steps ← generate_steps  //生成归结步骤
14.                              return steps            //找到矛盾
15.      if not next_generation then 
             break
16.      generations ← next_generation
17.  return steps

//寻找两个子句中原子公式互补对
function find_formula_pair(clause1,clause2)
    inputs: clause1,clause2         //两个子句
    returns: (formula1,formula2)    //互补对或(null, null)
1.   for (formula1,formula2) in (clause1,clause2) do
2.       if complementary_pair(formula1,formula2) then //如果f1和f2互补
            return (formula1,formula2)
3.   return (null,null)

//归结操作
function resolve(clause1,clause2,formula1,formula2)
    inputs: clause1,clause2     //两个子句
           formula1,formula2    //要消去的互补对
    returns: new_clause         //归结后的新子句
1.  return (clause1 ∪ clause2)-{formula1,formula2} //集合的差集操作
```

# **Algorithm 2: MGU (Most General Unify)**

```python
Algorithm 2: MGU (Most General Unify)
function MGU(formula1,formula2)
    inputs: formula1, formula2  // 要合一的原子公式
    outputs: σ                  // 最一般合一替换集
1.   D ← []                     // 初始化差异集
2.   σ ← {}                     // 初始化替换集
3.   while true do
4.       D ← difference(formula1, formula2) //difference函数得到差异集
         //D中的元素有可能为原子公式，个体变量，个体常量
5.       if D == [] then    //不存在差异集退出
            break  
6.       for items in D do  //遍历D中的元素,令s为变量类型，t可以是任何形式
7.           if items is variable then
8.               s ← items,t ← another
9.       if (s is variable) ∧ (s is not in t) then//s是变量且不在t中出现
10.          σ ← σ ∘ {s:t}  //使得置换{s:t}与σ进行复合
11.      else then          //其他情况下直接退出
             break
12.      formula1 ← formula1·σ  //应用置换集
13.      formula2 ← formula2·σ
14.  return σ

//计算差异集
function difference(formula1, formula2)
    inputs: formula1, formula2  //要比较的两个原子公式
    outputs: D                  //差异集
1.   pred1, terms1 ← extrct(formula1)
2.   pred2, terms2 ← extrct(formula2)   
3.   if pred1 ≠ pred2 then
         return [formula1, formula2]  
4.   if len(terms1) ≠ len(terms2) then
5.      return [formula1, formula2]
6.   for i ← 1 to len(terms1) do
7.       if difference(terms1[i],terms2[i]) then
8.           differences.append(difference(terms1[i],terms2[i]))   
9.   return differences[0]

//提取原子公式的谓词和项的原子公式
function extrct(formula)
    inputs: formula //需要提取谓词和项的原子公式
    outputs: predicate,terms  //谓词和项
1.   left ← formula.find('(') 
2.   right ← formula.rfind(')') 
     //单个变量无谓词返回空谓词以及自己本身就是自己唯一的项
3.   if left == -1 or right == -1 then  
        return '',formula
     //若有否定词,则从第二个字符到左括号的位置为谓词
4.   if formula.startswith('~') then  
5.      predicate ← formula[1:left]
     //其他情况从第一个字符到左括号为谓词
6.   else then                      
7.        predicate ← formula[0:left]
     //将项字符串按照逗号分割转换成列表
8.   terms ← [term for term in formula[left+1:right].split(',')]
9.   return predicate,terms
```

# **Agorithm 3: First Order Logic Resolution**

```python
Agorithm3: First Order Logic Resolution
function FirstOrderLogicResolution(KB)
    inputs:KB       //一阶逻辑子句集合
    returns: steps  //归结步骤列表

1. var clause_dict:dict
   var clauses,steps:list
   var generations//记录归结代数
   var parent_info:dict  //记录子句亲本信息
2. while true do
3. for pre_generation in generations do
4. for clause1 in current_generation do
5. for clause2 in pre_generation do
   //当前代一次与之前的所有代一一配对归结
   //获得两个子句的所有可能的合一置换集
6. substitution_ways ← MGU_clause(clause1,clause2)
   //应用置换集
7. for σ in substitution_ways do
8. new_clause1,new_clause2 ← clause1·σ,clause2·σ
   //寻找互补对
9. formula1,formula2 ← find_formula_pair
10. if formula1,formula2 then  //如果找到互补对
    //归结
11. new_clause ← resolve(new_clause1,new_clause2)
12. if new_clause not in clauses then //如果是新子句
    //更新新子句和归结步骤记录亲本信息
13. update(clauses,steps,parent_info)
14. if new_clause == () then   //如果是空子句
    //生成归结路径
15. resolution_path ← generate_resolution_path
16. steps ← generate_steps  //生成归结步骤
17. return steps            //找到矛盾
18. if not next_generation then
    break
19. generations ← next_generation
20. return steps
    other function
    //生成归结路径
    function generate_resolution_path(new_clause, parent_info) → resolution_path
    inputs: new_clause      //空子句
    parent_info     //亲本信息
    returns: resolution_path //归结路径
    //生成步骤信息
    function generate_steps(resolution_path, clause_dict, parent_info, KB) → steps
    inputs: resolution_path  //归结路径
    clause_dict      //子句编号字典
    parent_info      //亲本信息
    KB              //初始子句集
    returns: steps          //步骤列表
    //寻找两个子句中原子公式互补对
    function find_formula_pair(clause1,clause2) → (formula1,formula2)
    inputs: clause1,clause2         //两个子句
    returns: (formula1,formula2)    //互补对或(null, null)
    //归结操作
    function resolve(clause1,clause2,formula1,formula2) → new_clause
    inputs: clause1,clause2     //两个子句
    formula1,formula2    //要消去的互补对
    returns: new_clause         //归结后的新子句
    //计算子句的所有可能的置换集
    function MGU_clause(clause1, clause2) → list of substitutions
    inputs: clause1, clause2         //两个子句
    returns: list of substitutions   //所有可能的MGU置换
    //应用置换集
    function replace_clause(clause,σ) → replaced_clause
    inputs: clause      //要处理的子句
    σ           //要应用的替换集
    returns: new_clause
```
