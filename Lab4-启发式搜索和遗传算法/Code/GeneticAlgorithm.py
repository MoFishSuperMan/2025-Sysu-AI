import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import *
plt.rcParams['font.sans-serif'] = ['SimHei']    #使用黑体字体
plt.rcParams['axes.unicode_minus'] = False      #解决负号显示问题

class GeneticAlgTSP():
    def __init__(self,filename,population_size=100,muta_rate=0.5,cross_func=1):
        # 类成员变异率的初始化
        self.muta_rate=muta_rate
        # 类成员种群大小的初始化
        self.population_size=population_size
        # 类成员城市坐标的初始化
        self.cities=self.read_tsp(filename)
        # 初代种群的初始化
        self.population=self.init_population(population_size)
        # 城市图的距离矩阵
        self.distance_matrix=self.calculate_distance(self.cities)
        # 每一次迭代的较优解和较短距离
        self.generation_best_choice=[]
        self.generation_best_fitness=[]
        self.cross_func=cross_func
    
    # 读取文件函数
    def read_tsp(self,filename):
        cities=[]
        with open(filename,'r') as file:
            coord_label=False
            for line in file.readlines():
                if coord_label:
                    # 将一整行字符串用逗号分开在转换成实数存到cities中
                    city=line.strip().split(' ')
                    cities.append([float(city[1]),float(city[2])])
                # 文件中开始读取的标志
                if line.startswith('NODE_COORD_SECTION'):
                    coord_label=True
        return np.array(cities)
    
    def init_population(self,population_size):
        '''初始化初代种群使用随机组合的方式'''
        population=[]
        cities_size=len(self.cities)
        for _ in range(population_size):
            population.append(np.random.permutation(cities_size))
        return np.array(population)
    
    def calculate_distance(self, cities):
        '''计算城市间距离的距离矩阵'''
        # 使用np中广播机制一次性计算距离矩阵
        diff=cities[:,np.newaxis]-cities[np.newaxis,:]
        distance_matrix=np.linalg.norm(diff,axis=2)
        return distance_matrix
    
    def fitness(self, individual):
        '''计算个体的适应度'''
        # 使用向量化操作计算适应度
        distances=self.distance_matrix[individual[:-1],individual[1:]]
        value=np.sum(distances)+self.distance_matrix[individual[-1],individual[0]]
        return 1/value

    # 选择个体的方式
    def tournament_selection(self):
        '''随机竞争锦标赛选择法'''
        # 锦标赛规模，即每次从种群中选取的个体数量，可以根据需要调整
        tournament_size = 9 
        fitnesses = np.array([self.fitness(individual) for individual in self.population])
        selected_indices = []
        for _ in range(2):  
            # 从种群中随机选取个体组成锦标赛
            tournament_indices = np.random.choice(len(fitnesses),tournament_size)  
            # 获取锦标赛中个体的适应度
            tournament_fitnesses = fitnesses[tournament_indices]  
            # 找到锦标赛中适应度最高的个体的索引
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]  
            selected_indices.append(winner_index)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]
    def roulette_selection(self):
        '''轮盘赌'''
        # 依据适应度占总适应度的比例为概率选择个体
        fitnesses=np.array([self.fitness(individual) for individual in self.population])
        total_fitness=sum(fitnesses)
        index1,index2=np.random.choice(len(fitnesses),size=2,p=fitnesses/total_fitness)
        return self.population[index1],self.population[index2]

    def selection(self):
        return self.tournament_selection()

    # 交叉互换的方式
    def PMX_cross(self,parents):
        parent1,parent2=parents
        child1,child2=np.copy(parent1),np.copy(parent2)
        # 随机选择两个位置作为交叉区域的起始和结束位置
        start,end=sorted(np.random.randint(0,len(child1),2))
        # 进行部分映射交叉操作，交换两个子代在交叉区域内的基因片段
        child1[start:end+1],child2[start:end+1]=parent1[start:end+1],parent2[start:end+1]
        # 解决交换后的冲突问题，采用映射的方式，使得个体合理化
        for index in range(len(child1)):
            if index < start or index > end:
                while child1[index] in child1[start:end+1]:
                    match_index=np.where(child1[start:end+1] == child1[index])[0]
                    if match_index.size > 0:
                        child1[index]=child2[match_index[0]+start]
                while child2[index] in child2[start:end+1]:
                    match_index=np.where(child2[start:end+1] == child2[index])[0]
                    if match_index.size > 0:
                        child2[index]=child1[match_index[0]+start]
        return child1,child2
    def OX_Cross(self,parents):
        parent1,parent2=parents
        start,end=sorted(np.random.randint(0,len(parent1),2))
        child1=np.zeros(len(parent1),dtype=int)
        child2=np.zeros(len(parent2),dtype=int)
        child1[start:end+1]=parent1[start:end+1]
        child2[start:end+1]=parent2[start:end+1]
        index1,index2=0,0
        for i in range(len(parent2)):
            gene=parent2[i]
            if index1 == start:
                index1+=end-start+1
            if gene not in child1[start:end + 1]:
                child1[index1]=gene
                index1+=1
        for i in range(len(parent2)):
            gene=parent1[i]
            if index2 == start:
                index2+=end-start+1
            if gene not in child2[start:end + 1]:
                child2[index2]=gene
                index2+=1
        return child1,child2
    def PBX_Cross(self,parents):
        parent1, parent2 = parents
        index_num = np.random.randint(0, len(parent1))
        indexs = sorted(np.random.randint(0, len(parent1), index_num))
        child1 = np.full(len(parent1),-1,dtype=int)
        child2 = np.full(len(parent1),-1,dtype=int)
        child1[indexs] = parent1[indexs]
        child2[indexs] = parent2[indexs]
        remaining_gene_index1 = 0
        for i in range(len(parent1)):
            if i not in indexs:
                while parent2[remaining_gene_index1] in child1:
                    remaining_gene_index1 += 1
                child1[i] = parent2[remaining_gene_index1]
                remaining_gene_index1 += 1
        remaining_gene_index2 = 0
        for i in range(len(parent1)):
            if i not in indexs:
                while parent1[remaining_gene_index2] in child2:
                    remaining_gene_index2 += 1
                child2[i] = parent1[remaining_gene_index2]
                remaining_gene_index2 += 1
        return child1, child2

    def crossover(self,parents):
        '''交叉互换产生新个体'''
        cross_ways=[self.PMX_cross,self.OX_Cross,self.PBX_Cross]
        cross_way=random.choice(cross_ways)
        child1,child2=cross_ways[self.cross_func](parents)
        return child1,child2

    # 变异的方式
    def inversion_mutation(self,individual):
        '''倒置变异'''
        # 选择一个子序列将其倒置
        point1,point2=sorted(np.random.randint(0,len(individual),2))
        offspring=np.copy(individual)
        offspring[point1:point2+1]=offspring[point1:point2+1][::-1]
        return offspring
    def insertion_mutation(self,individual):
        '''插入变异'''
        # 选一个数字移动到另一个位置
        src,pos=np.random.randint(0,len(individual),2)
        tem=np.concatenate((individual[:src],individual[src+1:]))
        offspring=np.concatenate((tem[:pos],[individual[src]],tem[pos:]))
        return offspring
    def displacement_mutation(self,individual):
        '''位移变异'''
        # 选择一段移动到另一个位置
        point1,point2,pos=sorted(np.random.randint(0,len(individual),3))
        tem=np.concatenate((individual[:point1],individual[point2+1:]))
        offspring=np.concatenate((tem[:pos],individual[point1:point2+1],tem[pos:]))
        return offspring
    def swap_mutation(self,individual):
        '''交换变异'''
        # 选择两个基因交换位置
        point1,point2=np.random.randint(0,len(individual),2)
        offspring=np.copy(individual)
        offspring[point1],offspring[point2]=offspring[point2],offspring[point1]
        return offspring

    def mutation(self,individual):
        '''新个体可能发生变异'''
        if np.random.rand() > self.muta_rate:
            return individual
        else:
            mutation_ways=[self.inversion_mutation,self.insertion_mutation,self.displacement_mutation,self.swap_mutation]
            way=random.choices(mutation_ways,weights=[0.6,0.05,0.3,0.05])[0]
            #way=random.choice(mutation_ways)
            offspring=way(individual)
            return offspring

    def iterate(self,num_iterations):
        self.num_interations=num_iterations
        for i in trange(0,num_iterations):#trange
            childs=[]
            for _ in range(self.population_size//2):
                # 从当前种群中选两个亲本个体
                selection=self.selection()
                # 亲本个体杂交产生新的个体
                child1,child2=self.crossover(selection)
                # 产生子代可能发生变异
                child1,child2=self.mutation(child1),self.mutation(child2)
                # 将新产生的个体加入到childs
                childs.extend([child1,child2])
            # 选择新一代的种群
            new_population=np.concatenate((np.array(childs),self.population))
            # 根据适应度最大的前100个构成下一个种群(优胜劣汰)
            fitness_values = np.array([self.fitness(individual) for individual in new_population])
            top_indices = np.argpartition(-fitness_values,self.population_size)[:self.population_size]
            new_population = new_population[top_indices]
            # 记录当代的最优解和适应度
            best_individual = new_population[np.argmax(fitness_values[top_indices])]
            self.generation_best_choice.append(best_individual)
            self.generation_best_fitness.append(1/self.fitness(best_individual))
            #更新新一代的种群
            self.population=new_population
        return self.generation_best_choice[-1]

    def draw_result(self):
        interation=[0,
                    (len(self.generation_best_choice)-1)//4,
                    (len(self.generation_best_choice)-1)//2,
                    len(self.generation_best_choice)-1]
        plt.figure(figsize=(10,6))
        # 迭代进度0%，25%，50%，100%时刻的最优路线图
        for n,index in enumerate(interation):
            route=self.generation_best_choice[index]
            plt.subplot(2,2,n+1)
            plt.scatter(*self.cities.T,c='red',marker='o',s=10)
            #for i,(x,y) in enumerate(self.cities):
            #    plt.text(x,y+10,str(i+1),fontsize=3,ha='center',va='center')
            for i in range(-1,len(route)-1):
                plt.plot([self.cities[route[i]][0],self.cities[route[i+1]][0]],
                        [self.cities[route[i]][1],self.cities[route[i+1]][1]],'b-')
            plt.xticks([])
            plt.yticks([])
            plt.title(f'第{index+1}代最优路线图' if n!=3 else '最后一代最优路线图')
            plt.text(0.5,0.96, f'适应度: {self.generation_best_fitness[index]:.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=10)
        plt.tight_layout()
        plt.show()
        # 每一代的最优适应度随迭代次数的变化
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(self.generation_best_fitness)+1),self.generation_best_fitness)
        plt.xlabel('迭代次数')
        plt.xticks(rotation=45)
        plt.ylabel('最优适应度')
        plt.title('每一代的最优适应度随迭代次数的变化')
        plt.grid(True)
        plt.show()

def difference_population_size_result(test_tsp):
    '''不同population_size的效果'''
    population_sizes=[100,200,300,400]
    plt.figure(figsize=(10,6))
    for n,population_size in enumerate(population_sizes):
        np.random.seed(100)
        plt.subplot(2,2,n+1)
        tsp_solution=GeneticAlgTSP(test_tsp,population_size=population_size)
        tsp_solution.iterate(100)
        route=tsp_solution.generation_best_choice[-1]
        plt.scatter(*tsp_solution.cities.T,c='red',marker='o',s=10)
        for i in range(-1,len(route)-1):
                plt.plot([tsp_solution.cities[route[i]][0],tsp_solution.cities[route[i+1]][0]],
                        [tsp_solution.cities[route[i]][1],tsp_solution.cities[route[i+1]][1]],'b-')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'每代种群数为 {population_size} 迭代100代的当前最优结果图')
        plt.text(0.5,0.96, f'适应度: {tsp_solution.generation_best_fitness[-1]:.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=10)
        print(f'当每代种群数为 {population_size} 时，迭代结束时最优解的路径长度为：{tsp_solution.generation_best_fitness[-1]}')
        print()
    plt.tight_layout()
    plt.show()

def difference_muta_rate_result(test_tsp):
    muta_rates=[0.01,0.1,0.5,0.9]
    plt.figure(figsize=(10,6))
    for n,muta_rate in enumerate(muta_rates):
        np.random.seed(100)
        plt.subplot(2,2,n+1)
        tsp_solution=GeneticAlgTSP(test_tsp,population_size=100,muta_rate=muta_rate)
        tsp_solution.iterate(100)
        route=tsp_solution.generation_best_choice[-1]
        plt.scatter(*tsp_solution.cities.T,c='red',marker='o',s=10)
        for i in range(-1,len(route)-1):
                plt.plot([tsp_solution.cities[route[i]][0],tsp_solution.cities[route[i+1]][0]],
                        [tsp_solution.cities[route[i]][1],tsp_solution.cities[route[i+1]][1]],'b-')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'变异率为 {muta_rate} 迭代100代的当前最优结果图')
        plt.text(0.5,0.96, f'适应度: {tsp_solution.generation_best_fitness[-1]:.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=10)
        print(f'当变异率为 {muta_rate} 时，迭代结束时最优解的路径长度为：{tsp_solution.generation_best_fitness[-1]}')
        print()
    plt.tight_layout()
    plt.show()

def difference_cross_func_result(test_tsp):
    functs=[0,1,2]
    name=["PMX",'OX','PBX']
    plt.figure(figsize=(10,6))
    for n,funct in enumerate(functs):
        np.random.seed(100)
        plt.subplot(2,2,n+1)
        tsp_solution=GeneticAlgTSP(test_tsp,population_size=100,cross_func=funct)
        tsp_solution.iterate(100)
        route=tsp_solution.generation_best_choice[-1]
        plt.scatter(*tsp_solution.cities.T,c='red',marker='o',s=10)
        for i in range(-1,len(route)-1):
                plt.plot([tsp_solution.cities[route[i]][0],tsp_solution.cities[route[i+1]][0]],
                        [tsp_solution.cities[route[i]][1],tsp_solution.cities[route[i+1]][1]],'b-')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'{name[n]} 迭代100代的当前最优结果图')
        plt.text(0.5,0.96, f'适应度: {tsp_solution.generation_best_fitness[-1]:.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                fontsize=10)
        print(f'{name[n]} 时，迭代结束时最优解的路径长度为：{tsp_solution.generation_best_fitness[-1]}')
        print()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    filename1="dj38.tsp"    #6656
    filename2='qa194.tsp'   #9352
    filename3='uy734.tsp'   #79114
    #filename4='mu1979.tsp'  #86891
    tsp_solution=GeneticAlgTSP(filename3)
    print(tsp_solution.iterate(10000))
    #tsp_solution.draw_result()
    #difference_population_size_result(filename1)
    #difference_muta_rate_result(filename2)
    #difference_cross_func_result(filename1)


