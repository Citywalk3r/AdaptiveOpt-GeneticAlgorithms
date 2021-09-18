from pathlib import Path
import numpy as np
from numpy.core.fromnumeric import trace
from genetic import ga
from functools import reduce


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



    

    def eval_func(self, currState):
        """https://en.wikipedia.org/wiki/Quadratic_assignment_problem
            score = trace(W * X * D * X_transp)
        """
        x_t = generate_sq_tbl(currState)
        score = np.trace(reduce(np.dot, [self.flow_sq, x_t, self.dist_sq, x_t.T]))/2
        # print(score)
        return score
        
    
    def solve_qap(self, init_state=None, stop_crit_dict=None):
        """
        Solves the qap problem.
        """

        # list of historical populations
        generation_eval_list = []

        if self.is_debug:
            print("Flow table: \n{}".format(self.flow))
            print("Distance table: \n{}".format(self.dist))


        
        generation_list = ga(p_cross=0.9, p_mut=0.1, p_size=100, g_max=150, eval_f= self.eval_func)

        generation_eval_list = [[self.eval_func(individual) for individual in generation] for generation in generation_list]
        # print(len(generation_eval_list))
        # print(generation_eval_list)
        gen_eval_list_np = np.array(generation_eval_list)
          
        # Using numpy sum
        res = np.min(gen_eval_list_np, 1)
        min_of_gen_mins = np.min(res)
        print(res)
        print(len(res))

        # last_gen = generation_list[-1]
        # last_gen_eval = [self.eval_func(individual) for individual in last_gen]

        # print("Best solution: ",min(last_gen_eval))
        print("Best solution: ",min_of_gen_mins)

        import matplotlib.pyplot as plt

        plt.plot(range(len(res)), res)
        plt.xlabel("generations")
        plt.ylabel("best individual score")
        plt.show()

            
if __name__ == "__main__":
    QAP = QAP(is_debug=False)
    QAP.solve_qap()