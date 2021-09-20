import numpy as np
from genetic_algorithm import ga
import matplotlib.pyplot as plt
import math
import random
import csv

class ACKLEY:

    def __init__(self, is_debug):
        self.is_debug = is_debug
    
    def generate_init_pop(self, p_size, seed):
        """
        Generates the initial population given population size.
        """
        np.random.seed(seed=seed)
        init_generation = [np.ndarray.tolist(np.random.uniform(-32,32,2)) for _ in range(p_size)]
        # print(init_generation)
        return init_generation
    
    def tournament_selection(self, population,  eval_f):
        """
        To find the first parent, selects 3 parents from the population and returns the best.
        To find the second parent, selects k parents from the population and returns the best.
        """
        k=40
        parent1 = random.sample(population, 2)
        parent1 = sorted(parent1, key=lambda parent: eval_f(parent))

        parent2 = random.sample(population, k)
        parent2 = sorted(parent2, key=lambda parent: eval_f(parent))
        return parent1[0], parent2[0]
    

    def single_point_crossover(self, p1, p2):
        """
        Implements the crossover function.
        Single point crossover with repair for infeasible solutions.
        """

        modifier = np.random.normal(loc=0.0, scale=0.001)

        dim = np.random.randint(2)

        M = np.random.randint(1,1001)

        delta1 = p1[dim]/M
        delta2 = p2[dim]/M

        child1 = p1.copy()
        child2 = p2.copy()

        if (p1[dim] + delta2 - delta1 <= 32 and p1[dim] + delta2 - delta1 >= -32 ):
            child1[dim] = p1[dim] + delta2 - delta1

        if (p2[dim] - delta2 + delta1 <= 32 and p2[dim] - delta2 + delta1 >= -32 ):
            child2[dim] = p2[dim] - delta2 + delta1

        if self.is_debug:
            print("Parent 1: ", p1)
            print("Parent 2: ", p2)

            print("modifier: ", modifier)

            print("Child 1 : ", child1)
            print("Child 2 : ", child2)
        
        return child1, child2

    def mutate_degrading(self, child, p_mut, g_max, curr_g):
        """Implements the mutation operator.
           Picks 2 random elements from the array and swaps them.
        """
        
        p_mut_modifier = g_max/curr_g
        lower, upper = 0, p_mut
        p_mut_mod = lower + (upper - lower) * p_mut_modifier

        # print("p_mut: ",p_mut_mod)

        p = np.random.uniform(0,1)

        if p <= p_mut_mod:
            dim = np.random.randint(2)
            M = np.random.randint(1,11)
            deltaX = (32 - (-32))/M
            k = 1 - 1/p_mut_modifier
            # print("k: ",k)
            k_sign = np.random.randint(2)
            if k_sign == 0:
                if child[dim] + k*deltaX <=32:
                    child[dim] += k*deltaX
            else:
                if child[dim] - k*deltaX >=-32:
                    child[dim] -= k*deltaX
        return child



    def eval_func(self, individual):
        """
        Evaluates the current state by
        calculating the function result.
        """

        x = individual[0]
        y = individual[1]
        f = -20 * math.exp(-0.2 * math.sqrt(0.5*(x**2+y**2))) - math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y))) + 20 + math.e
        return f
    
    def solve_ackley(self):
        """
        Calls the GA for the Ackley function problem and plots the results.
        """
      
        fig, axs = plt.subplots(5, 6)

        p_cross = 0.9
        p_mut_list = [0.9, 0.5, 0.1]
        p_size_list = [50, 100, 150]
        g_max_list = [300, 400, 500]

        colors=['xkcd:light cyan','xkcd:light seafoam','xkcd:very light blue','xkcd:eggshell','xkcd:very light purple',
        'xkcd:light light blue','xkcd:lightblue','xkcd:buff','xkcd:light mint','xkcd:light periwinkle',]

        with open('../results_ackley2.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["gen_max","pop_size","mut_prob","0","1","2","3","4","5","6","7","8","9"])

            for g_max in g_max_list:
                for p_idx, p_size in enumerate(p_size_list):
                    for mut_p in p_mut_list:
                        write_string = [g_max, p_size, mut_p]
                        for i in range(10):
                            generation_list = ga(ipop_f=self.generate_init_pop,
                                                p_cross=p_cross,
                                                p_mut=mut_p, p_size=p_size,
                                                g_max=g_max,
                                                eval_f= self.eval_func,
                                                seed=i,
                                                selection_f=self.tournament_selection,
                                                crossover_f=self.single_point_crossover,
                                                mutation_f=self.mutate_degrading)


                            generation_eval_list = [[self.eval_func(individual) for individual in generation] for generation in generation_list]
                            gen_eval_list_np = np.array(generation_eval_list)
                    
                            # list of mins of each generation
                            gen_mins = np.min(gen_eval_list_np, 1)
            
                            # best solution
                            best = np.min(gen_mins)
                            write_string.append(best)


                        writer.writerow(write_string)

        #             axs.flat[i*3 + p_idx].plot(range(g_max), gen_mins, label="p_mut: "+str(mut_p))

        #         best_per_solution = np.min(best_per_pmut)

        #         plot_idx = i*3 + p_idx
        #         axs.flat[plot_idx].set_title('s={:},p_size={:},g_max={:}'.format(i,p_size,g_max))
        #         axs.flat[plot_idx].set(xlabel="generations", ylabel="best individual score")
        #         axs.flat[plot_idx].hlines(y=best_per_solution, xmin=0, xmax=g_max, linewidth=2, color='r', label="best: "+str(best_per_solution))
        #         axs.flat[plot_idx].legend()
        #         axs.flat[plot_idx].label_outer()
        #         axs.flat[plot_idx].set_facecolor(colors[i])
        
        # plt.show()


        # p_cross = 0.9
        # p_mut = 0.9
        # p_size = 200
        # g_max = 300

        # generation_list = ga(ipop_f=self.generate_init_pop,
        #                                 p_cross=p_cross,
        #                                 p_mut=p_mut, p_size=p_size,
        #                                 g_max=g_max,
        #                                 eval_f= self.eval_func,
        #                                 seed=0,
        #                                 selection_f=self.tournament_selection,
        #                                 crossover_f=self.single_point_crossover,
        #                                 mutation_f=self.mutate_degrading)


        # generation_eval_list = [[self.eval_func(individual) for individual in generation] for generation in generation_list]
        # gen_eval_list_np = np.array(generation_eval_list)
    
        # # list of mins of each generation
        # gen_mins = np.min(gen_eval_list_np, 1)
        # # print(generation_eval_list)
        # # print(gen_mins)

        # # best solution
        # best = np.min(gen_mins)
        # print("Best solution: ", best)

        
        # plt.title('p_size={:},g_max={:}'.format(p_size,g_max))
        # plt.axes(xlabel="generations", ylabel="best individual score")
        # plt.plot(range(g_max), gen_mins)
        # plt.hlines(y=best, xmin=0, xmax=g_max, linewidth=2, color='r', label="best: "+str(best))
        # plt.legend()
        # plt.show()
    


if __name__ == "__main__":
    ACKLEY = ACKLEY(is_debug=False)
    ACKLEY.solve_ackley()