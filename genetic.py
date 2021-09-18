import itertools
import random
import math
import numpy as np


def generate_init_pop(p_size):
    init_generation = [np.random.permutation(range(1,16)) for _ in range(p_size)]

    # p1 = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # eval 746
    # p2 = np.asarray([10,12,8,9,11,6,5,3,13,1,15,4,14,2,7]) # eval 647
    # p3 = np.asarray([9,8,13,2,12,11,7,14,3,4,1,5,6,15,10]) # eval 663
    # p4 = np.asarray([10,5,8,7,12,9,6,13,2,4,15,3,14,1,11]) # eval 702
    # p5 = np.asarray([7,10,14,2,1,12,8,6,4,13,5,15,11,3,9]) # eval 713

    # generation = [p1,p2,p3,p4,p5]
    return init_generation



def tournament_selection(generation, k,  eval_f):
    parents = random.sample(generation, k)
    # print(parents)
    parents = sorted(parents, key=lambda parent: eval_f(parent))
    # print(parent1,eval_f(parent1))
    return parents[0], parents[1]


def biased_roulette_wheel():
    pass

def crossover(p1, p2):
    """Implements the crossover function.
        Single point crossover.
    """
    # generating the random number to perform crossover
    k = random.randint(0, 15)
    # print("Crossover point :", k)

    elements = list(range(1,16))
    random.shuffle(elements)

    child1 = np.array(p1, copy=True) 
    child2 = np.array(p2, copy=True) 

    np.put(child1, range(k,15), p2[k:])
    np.put(child2, range(k,15), p1[k:])

    # print("Parent 1: ", p1)
    # print("Parent 2: ", p2)

    # print("Child 1 : ", child1)
    # print("Child 2 : ", child2)

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

    # print("Elements occuring more than once c1:")
    for i in res:
        # print(i)
        k = random.randint(0, 1)
        for e in elements:
            if e not in child1:
                # print("Element {} will be placed in {} position of child 1".format(e,i[k]))
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

    # print("Elements occuring more than once c2:")
    for i in res:
        # print(i)
        k = random.randint(0, 1)
        for e in elements:
            if e not in child2:
                # print("Element {} will be placed in {} position of child 2".format(e,i[k]))
                child2[i[k]] = e
                break
    
    # print("Fixed child 1: ",child1)
    # print("Fixed child 2: ",child2)


    return child1, child2

def mutate(child, p_mut):
        """Implements the mutation operator.
           Picks 2 random elements from the array and swaps them.
        """
        mutated_child = np.array(child, copy=True)

        # print("~~~~~~~~~~~~ Begin child mutation ~~~~~~~~")
        # print("Child shape: ", child.shape)

        it = np.nditer(mutated_child, flags=['f_index'])
        for x in it:
            # print("~~~~~~~~~~~~ Currently at allele: ", it.index)
            if random.uniform(0, 1) <= p_mut:
                child_without_allele = np.delete(mutated_child, it.index)
                # print(mutated_child)
                # print(child_without_allele)
                random_element_idx = np.random.choice(child_without_allele.size, size=1, replace=False)
                idx_to_replace_with = np.where(mutated_child == child_without_allele[random_element_idx[0]])
                # print("Element to replace", mutated_child[it.index])
                # print("Replacing with: ", mutated_child[idx_to_replace_with[0][0]])

                # Swap the selected elements
                # mutated_child[it.index], mutated_child[idx_to_replace_with[0][0]] = mutated_child[idx_to_replace_with], mutated_child[it.index]

                element_to_swap = mutated_child[it.index]
                element_to_swap2 = mutated_child[idx_to_replace_with[0][0]]

                np.put(mutated_child, it.index, element_to_swap2)
                np.put(mutated_child, idx_to_replace_with[0][0], element_to_swap)

                # print("b4 Mutation: ", child)
                # print("After Mutation: ", mutated_child)

        return mutated_child

def ga(p_cross=1, p_mut=0.1, p_size=6, g_max=5, eval_f= None):

        """Genetic algorithm

        Parameters:
            p_cross : probability to produce 2 children at crossover
            p_mut : mutation probability (for each allele)
            p_size : population size
            g_max : max number of generations
            eval_f : evaluation function
            crossover_f : crossover function
            mutate_f : mutation function

        Returns:
            x_final : final state array
            state_list : list of historical states

        """

        print("Running the genetic algorithm...")

        # region initialization

        # final populations for each generation (used for visualization)
        generation_list = []
        g = 1
        generation = generate_init_pop(p_size)
        generation_list.append(generation)

        # endregion

        while (g!=g_max):
            # print("Generation: ",g)
            # print("Population size: ",len(generation))
            children_list = []
            for _ in range (p_size//2):
                p1, p2 = tournament_selection(generation,p_size//5,eval_f)
                if random.uniform(0, 1) <= p_cross:
                    c1, c2 = crossover(p1, p2)
                    c1 = mutate(c1, p_mut)
                    c2 = mutate(c2, p_mut)
                    children_list.extend([c1,c2])
                else:
                    children_list.extend([p1,p2])
            generation = children_list
            generation_list.append(generation)
            g+=1
        return generation_list