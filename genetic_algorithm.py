import itertools
import random
import math
import numpy as np

def ga(ipop_f=None,p_cross=1, p_mut=0.1, p_size=6, g_max=5, eval_f= None, seed=0, selection_f=None, crossover_f=None, mutation_f=None ):

        """Genetic algorithm

        Parameters:
            ipop_f : initial population generation function
            p_cross : probability to produce 2 children at crossover
            p_mut : mutation probability (for each allele)
            p_size : population size
            g_max : max number of generations
            eval_f : evaluation function
            selection_f: parent selection function
            crossover_f : crossover function
            mutate_f : mutation function

        Returns:
            x_final : final state array
            generation_list : list of historical states

        """

        print("Running the genetic algorithm...")

        # region initialization

        # final populations for each generation (used for visualization)
        generation_list = []
        g = 1
        generation = ipop_f(p_size, seed)
        generation_list.append(generation)

        # endregion

        while (g!=g_max):
            # print("Generation: ",g)
            # print("Population size: ",len(generation))
            children_list = []
            for _ in range (p_size//2):
                p1, p2 = selection_f(generation,eval_f)
                if random.uniform(0, 1) <= p_cross:
                    c1, c2 = crossover_f(p1, p2)
                    c1 = mutation_f(child=c1, p_mut=p_mut, g_max=g_max, curr_g=g)
                    c2 = mutation_f(child=c2, p_mut=p_mut, g_max=g_max, curr_g=g)
                    children_list.extend([c1,c2])
                else:
                    children_list.extend([p1,p2])
            generation = children_list
            generation_list.append(generation)
            g+=1
        return generation_list