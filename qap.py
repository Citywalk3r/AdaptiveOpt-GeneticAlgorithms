from pathlib import Path
import numpy as np
import random
from genetic_algorithm import ga
from functools import reduce
import matplotlib.pyplot as plt
import csv
plt.rcParams.update({'font.size': 6})


def parse_data():
    data_file = Path("flow_dist_tbl.csv")
    try:
        data_file.resolve(strict=True)
    except FileNotFoundError:
        print ("flow_dist_tbl.csv not found. Please include the data file in the root folder. Aborting..\n")
        return
    else:
        data = np.genfromtxt(data_file, dtype=int, delimiter=',')
        return np.tril(data), np.triu(data)



def generate_sq_tbl(currState):
    """Generates 15x15 matrix with one-hot encoding of the current state.
        If department 15 has index 2, element (15,1) = 1

        Example: [10,12,8,9,11,6,5,3,13,1,15,4,14,2,7]

            becomes

            [[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
            [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
            [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
            [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
            [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
            [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
            [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]]
    """
    x_t = np.zeros((15,15), dtype=int)
    for index in range(len(currState)):
        x_t[currState[index]-1][index] = 1
    return x_t

class QAP:

    def __init__(self, is_debug):
        self.is_debug = is_debug
        flow, dist = parse_data()

        # 15x15 symmetric flow matrix
        self.flow_sq = flow + flow.T - np.diag(np.diag(flow))
        # 15x15 symmetric distance matrix
        self.dist_sq = dist + dist.T - np.diag(np.diag(dist))

    def generate_init_pop(self, p_size, seed):
        """
        Generates the initial population given population size.
        """
        np.random.seed(seed=seed)
        init_generation = [np.random.permutation(range(1,16)) for _ in range(p_size)]
        # print(init_generation)
        return init_generation

    def tournament_selection(self, population,  eval_f):
        """
        To find the first parent, selects 3 parents from the population and returns the best.
        To find the second parent, selects k parents from the population and returns the best.
        """
        k=5
        parent1 = random.sample(population, 3)
        parent1 = sorted(parent1, key=lambda parent: eval_f(parent))

        parent2 = random.sample(population, k)
        parent2 = sorted(parent2, key=lambda parent: eval_f(parent))
        return parent1[0], parent2[0]
    
    def single_point_crossover(self, p1, p2):
        """
        Implements the crossover function.
        Single point crossover with repair for infeasible solutions.
        """

        # Generate a random number as crossover point
        k = random.randint(0, 15)

        elements = list(range(1,16))
        random.shuffle(elements)

        child1 = np.array(p1, copy=True) 
        child2 = np.array(p2, copy=True) 

        np.put(child1, range(k,15), p2[k:])
        np.put(child2, range(k,15), p1[k:])

        if self.is_debug:
            print("Parent 1: ", p1)
            print("Parent 2: ", p2)

            print("Child 1 : ", child1)
            print("Child 2 : ", child2)

        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(child1)

        # sorts records array so all unique elements are together 
        sorted_records_array = child1[idx_sort]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])

        #filter them with respect to their size, keeping only items occurring more than once
        vals = vals[count > 1]
        res = filter(lambda x: x.size > 1, res)

        if self.is_debug:
            print("Elements occuring more than once c1:")

        for i in res:
            k = random.randint(0, 1)
            for e in elements:
                if e not in child1:
                    if self.is_debug:
                        print("Element {} will be placed in {} position of child 1".format(e,i[k]))
                    child1[i[k]] = e
                    break

        # print("Fixed child1: ",child1)

        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(child2)

        # sorts records array so all unique elements are together 
        sorted_records_array = child2[idx_sort]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])

        #filter them with respect to their size, keeping only items occurring more than once
        vals = vals[count > 1]
        res = filter(lambda x: x.size > 1, res)

        if self.is_debug:
            print("Elements occuring more than once c2:")
        for i in res:
            k = random.randint(0, 1)
            for e in elements:
                if e not in child2:
                    if self.is_debug:
                        print("Element {} will be placed in {} position of child 2".format(e,i[k]))
                    child2[i[k]] = e
                    break
        
        if self.is_debug:
            print("Fixed child 1: ",child1)
            print("Fixed child 2: ",child2)


        return child1, child2

    def mutate_2opt(self, child, p_mut):
        """Implements the mutation operator.
           Picks 2 random elements from the array and swaps them.
        """
        mutated_child = np.array(child, copy=True)

        it = np.nditer(mutated_child, flags=['f_index'])
        for x in it:

            if random.uniform(0, 1) <= p_mut:
                child_without_allele = np.delete(mutated_child, it.index)
                random_element_idx = np.random.choice(child_without_allele.size, size=1, replace=False)
                idx_to_replace_with = np.where(mutated_child == child_without_allele[random_element_idx[0]])

                if self.is_debug:
                    print("Element to replace", mutated_child[it.index])
                    print("Replacing with: ", mutated_child[idx_to_replace_with[0][0]])

                # Swap the selected elements
                element_to_swap = mutated_child[it.index]
                element_to_swap2 = mutated_child[idx_to_replace_with[0][0]]

                np.put(mutated_child, it.index, element_to_swap2)
                np.put(mutated_child, idx_to_replace_with[0][0], element_to_swap)

                if self.is_debug:
                    print("b4 Mutation: ", child)
                    print("After Mutation: ", mutated_child)

        return mutated_child

    
    def eval_func(self, individual):
        """https://en.wikipedia.org/wiki/Quadratic_assignment_problem
            score = trace(W * X * D * X_transp)
        """
        x_t = generate_sq_tbl(individual)
        score = np.trace(reduce(np.dot, [self.flow_sq, x_t, self.dist_sq, x_t.T]))/2
        # print(score)
        return score
        
    
    def solve_qap(self):
        """
        Calls the GA for the QAP problem and plots the results.
        """
        if self.is_debug:
            print("Flow table: \n{}".format(self.flow))
            print("Distance table: \n{}".format(self.dist))

        fig, axs = plt.subplots(5, 6)

        p_cross = 0.9
        p_mut_list = [0.1, 0.125, 0.075]
        p_size_list = [100, 160, 200]
        g_max = 5

        colors=['xkcd:light cyan','xkcd:light seafoam','xkcd:very light blue','xkcd:eggshell','xkcd:very light purple',
        'xkcd:light light blue','xkcd:lightblue','xkcd:buff','xkcd:light mint','xkcd:light periwinkle',]

        
        # with open('../results_qap_gen500.csv', 'w', newline='') as file:
            # writer = csv.writer(file)
            # writer.writerow(["Solution", "max_gen", "pop_size", "mut_prob", "best_score"])

        for i in range(10):
            for p_idx, p_size in enumerate(p_size_list):
                best_per_pmut = []
                for mut_p in p_mut_list:
                    generation_list = ga(ipop_f=self.generate_init_pop,
                                        p_cross=p_cross,
                                        p_mut=mut_p, p_size=p_size,
                                        g_max=g_max,
                                        eval_f= self.eval_func,
                                        seed=i,
                                        selection_f=self.tournament_selection,
                                        crossover_f=self.single_point_crossover,
                                        mutation_f=self.mutate_2opt)


                    generation_eval_list = [[self.eval_func(individual) for individual in generation] for generation in generation_list]
                    gen_eval_list_np = np.array(generation_eval_list)
                
                    # list of mins of each generation
                    gen_mins = np.min(gen_eval_list_np, 1)
        
                    # best solution
                    best = np.min(gen_mins)
                    best_per_pmut.append(best)

                    # writer.writerow([i, g_max, p_size, mut_p, best])

                    axs.flat[i*3 + p_idx].plot(range(g_max), gen_mins, label="p_mut: "+str(mut_p))

                best_per_solution = np.min(best_per_pmut)

                plot_idx = i*3 + p_idx
                axs.flat[plot_idx].set_title('s={:},p_size={:},g_max={:}'.format(i,p_size,g_max))
                axs.flat[plot_idx].set(xlabel="generations", ylabel="best individual score")
                axs.flat[plot_idx].hlines(y=best_per_solution, xmin=0, xmax=g_max, linewidth=2, color='r', label="best: "+str(best_per_solution))
                axs.flat[plot_idx].legend()
                axs.flat[plot_idx].label_outer()
                axs.flat[plot_idx].set_facecolor(colors[i])
        
        plt.show()
            
        

            
if __name__ == "__main__":
    QAP = QAP(is_debug=False)
    QAP.solve_qap()