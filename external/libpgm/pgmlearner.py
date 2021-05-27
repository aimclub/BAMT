# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the CyberPoint International, LLC nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
This module provides tools to generate Bayesian networks that are "learned" from a data set. The learning process involves finding the Bayesian network that most accurately models data given as input -- in other words, finding the Bayesian network that makes the data set most likely. There are two major parts of Bayesian network learning: structure learning and parameter learning. Structure learning means finding the graph that most accurately depicts the dependencies detected in the data. Parameter learning means adjusting the parameters of the CPDs in a graph skeleton to most accurately model the data. This module has tools for both of these tasks.

'''
import copy
import json
import itertools
try:
    import numpy as np
except ImportError:
    raise (ImportError, "numpy is not installed on your system.")

try: 
    from scipy.stats import chisquare
except ImportError:
    raise (ImportError, "scipy is not installed on your system.")


from external.libpgm.graphskeleton import GraphSkeleton
from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from external.libpgm.lgbayesiannetwork import LGBayesianNetwork
from external.libpgm.sampleaggregator import SampleAggregator

class PGMLearner():
    '''
    This class is a machine with tools for learning Bayesian networks from data. It contains the *discrete_mle_estimateparams*, *lg_mle_estimateparams*, *discrete_constraint_estimatestruct*, *lg_constraint_estimatestruct*, *discrete_condind*, *discrete_estimatebn*, and *lg_estimatebn* methods.

    '''
    def discrete_mle_estimateparams(self, graphskeleton, data):
        '''
        Estimate parameters for a discrete Bayesian network with a structure given by *graphskeleton* in order to maximize the probability of data given by *data*. This function takes the following arguments:

            1. *graphskeleton* -- An instance of the :doc:`GraphSkeleton <graphskeleton>` class containing vertex and edge data.
            2. *data* -- A list of dicts containing samples from the network in {vertex: value} format. Example::

                    [
                        {
                            'Grade': 'B',
                            'SAT': 'lowscore',
                            ...
                        },
                        ...
                    ]

        This function normalizes the distribution of a node's outcomes for each combination of its parents' outcomes. In doing so it creates an estimated tabular conditional probability distribution for each node. It then instantiates a :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` instance based on the *graphskeleton*, and modifies that instance's *Vdata* attribute to reflect the estimated CPDs. It then returns the instance. 

        The Vdata attribute instantiated is in the format seen in :doc:`unittestdict`, as described in :doc:`discretebayesiannetwork`.

        Usage example: this would learn parameters from a set of 200 discrete samples::

            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            from external.libpgm.pgmlearner import PGMLearner
            
            # generate some data to use
            nd = NodeData()
            nd.load("../tests/unittestdict.txt")    # an input file
            skel = GraphSkeleton()
            skel.load("../tests/unittestdict.txt")
            skel.toporder()
            bn = DiscreteBayesianNetwork(skel, nd)
            data = bn.randomsample(200)

            # instantiate my learner 
            learner = PGMLearner()

            # estimate parameters from data and skeleton
            result = learner.discrete_mle_estimateparams(skel, data)

            # output
            print json.dumps(result.Vdata, indent=2)

        '''
        assert (isinstance(graphskeleton, GraphSkeleton)), "First arg must be a loaded GraphSkeleton class."
        assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Second arg must be a list of dicts."

        # instantiate Bayesian network, and add parent and children data
        bn = DiscreteBayesianNetwork()
        graphskeleton.toporder()
        bn.V = graphskeleton.V
        bn.E = graphskeleton.E
        bn.Vdata = dict()
        for vertex in bn.V: 
            bn.Vdata[vertex] = dict()
            bn.Vdata[vertex]["children"] = graphskeleton.getchildren(vertex)
            bn.Vdata[vertex]["parents"] = graphskeleton.getparents(vertex)
            
            # make placeholders for vals, cprob, and numoutcomes
            bn.Vdata[vertex]["vals"] = []
            if (bn.Vdata[vertex]["parents"] == []):
                bn.Vdata[vertex]["cprob"] = []
            else:
                bn.Vdata[vertex]["cprob"] = dict()

            bn.Vdata[vertex]["numoutcomes"] = 0

        # determine which outcomes are possible for each node
        for sample in data:
            for vertex in bn.V:
                if (sample[vertex] not in bn.Vdata[vertex]["vals"]):
                    bn.Vdata[vertex]["vals"].append(sample[vertex])
                    bn.Vdata[vertex]["numoutcomes"] += 1

        # lay out probability tables, and put a [num, denom] entry in all spots:

        # define helper function to recursively set up cprob table
        def addlevel(vertex, _dict, key, depth, totaldepth):
            if depth == totaldepth:
                _dict[str(key)] = []
                for _ in range(bn.Vdata[vertex]["numoutcomes"]):
                    _dict[str(key)].append([0, 0])
                return
            else:
                for val in bn.Vdata[bn.Vdata[vertex]["parents"][depth]]["vals"]:
                    ckey = key[:]
                    ckey.append(str(val))
                    addlevel(vertex, _dict, ckey, depth+1, totaldepth)

        # put [0, 0] at each entry of cprob table
        for vertex in bn.V:
            if (bn.Vdata[vertex]["parents"]):
                root = bn.Vdata[vertex]["cprob"]
                numparents = len(bn.Vdata[vertex]["parents"])
                addlevel(vertex, root, [], 0, numparents)
            else:
                for _ in range(bn.Vdata[vertex]["numoutcomes"]):
                    bn.Vdata[vertex]["cprob"].append([0, 0])

        # fill out entries with samples:
        for sample in data:
            for vertex in bn.V:
                    
                # compute index of result
                rindex = bn.Vdata[vertex]["vals"].index(sample[vertex])

                # go to correct place in Vdata
                if bn.Vdata[vertex]["parents"]:
                    pvals = [str(sample[t]) for t in bn.Vdata[vertex]["parents"]]
                    lev = bn.Vdata[vertex]["cprob"][str(pvals)]
                else:
                    lev = bn.Vdata[vertex]["cprob"]

                # increase all denominators for the current condition
                for entry in lev:
                    entry[1] += 1

                # increase numerator for current outcome
                lev[rindex][0] += 1

        # convert arrays to floats
        for vertex in bn.V:
            if not bn.Vdata[vertex]["parents"]:
                bn.Vdata[vertex]["cprob"] = [x[0]/float(x[1]) for x in bn.Vdata[vertex]["cprob"]]
            else:
                for key in list(bn.Vdata[vertex]["cprob"].keys()):
                    try: 
                        bn.Vdata[vertex]["cprob"][key] = [x[0]/float(x[1]) for x in bn.Vdata[vertex]["cprob"][key]]
                        
                    # default to even distribution if no data points
                    except ZeroDivisionError:
                        bn.Vdata[vertex]["cprob"][key] = [1/float(bn.Vdata[vertex]["numoutcomes"]) for x in bn.Vdata[vertex]["cprob"][key]]

        # return cprob table with estimated probability distributions
        return bn

    def lg_mle_estimateparams(self, graphskeleton, data):
        '''
        Estimate parameters for a linear Gaussian Bayesian network with a structure given by *graphskeleton* in order to maximize the probability of data given by *data*. This function takes the following arguments:

            1. *graphskeleton* -- An instance of the :doc:`GraphSkeleton <graphskeleton>` class containing vertex and edge data.
            2. *data* -- A list of dicts containing samples from the network in {vertex: value} format. Example::

                    [
                        {
                            'Grade': 74.343,
                            'Intelligence': 29.545,
                            ...
                        },
                        ...
                    ]

        The algorithm used to calculate the linear Gaussian parameters is beyond the scope of this documentation -- for a full explanation, cf. Koller et al. 729. After the parameters are calculated, the program instantiates a :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` instance based on the *graphskeleton*, and modifies that instance's *Vdata* attribute to reflect the estimated CPDs. It then returns the instance. 

        The Vdata attribute instantiated is in the format seen in the input file example :doc:`unittestdict`, as described in :doc:`discretebayesiannetwork`.

        Usage example: this would learn parameters from a set of 200 linear Gaussian samples::

            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.lgbayesiannetwork import LGBayesianNetwork
            from external.libpgm.pgmlearner import PGMLearner
            
            # generate some data to use
            nd = NodeData()
            nd.load("../tests/unittestlgdict.txt")    # an input file
            skel = GraphSkeleton()
            skel.load("../tests/unittestdict.txt")
            skel.toporder()
            lgbn = LGBayesianNetwork(skel, nd)
            data = lgbn.randomsample(200)
        
            # instantiate my learner 
            learner = PGMLearner()

            # estimate parameters
            result = learner.lg_mle_estimateparams(skel, data)

            # output
            print json.dumps(result.Vdata, indent=2)

        '''
        assert (isinstance(graphskeleton, GraphSkeleton)), "First arg must be a loaded GraphSkeleton class."
        assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Second arg must be a list of dicts."

        # instantiate Bayesian network, and add parent and children data
        bn = LGBayesianNetwork()
        graphskeleton.toporder()
        bn.V = graphskeleton.V
        bn.E = graphskeleton.E
        bn.Vdata = dict()
        for vertex in bn.V: 
            bn.Vdata[vertex] = dict()
            bn.Vdata[vertex]["children"] = graphskeleton.getchildren(vertex)
            bn.Vdata[vertex]["parents"] = graphskeleton.getparents(vertex)
            
            # make placeholders for mean_base, mean_scal, and variance
            bn.Vdata[vertex]["mean_base"] = 0.0
            bn.Vdata[vertex]["mean_scal"] = []
            for parent in bn.Vdata[vertex]["parents"]:
                bn.Vdata[vertex]["mean_scal"].append(0.0)
            bn.Vdata[vertex]["variance"] = 0.0

        # make covariance table, array of E[X_i] for each vertex, and table
        # of E[X_i * X_j] for each combination of vertices
        cov = [[0 for _ in range(len(bn.V))] for __ in range(len(bn.V))]
        singletons = [0 for _ in range(len(bn.V))]
        numtrials = len(data)
        for sample in data:
            for x in range(len(bn.V)):
                singletons[x] += sample[bn.V[x]]
                for y in range(len(bn.V)):
                    cov[x][y] += sample[bn.V[x]] * sample[bn.V[y]]
        
        for x in range(len(bn.V)):
            singletons[x] /= float(numtrials)
            for y in range(len(bn.V)):
                cov[x][y] /= float(numtrials)

        # (save copy. this is the E[X_i * X_j] table) 
        product_expectations = [[cov[x][y] for y in range(len(bn.V))] for x in range(len(bn.V))] 

        for x in range(len(bn.V)):
            for y in range(len(bn.V)):
                cov[x][y] = cov[x][y] - (singletons[x] * singletons[y])
        
        # construct system of equations and solve (for each node)
        for x in range(len(bn.V)):
            
            # start with the E[X_i * X_j] table
            system = [[product_expectations[p][q] for q in range(len(bn.V))] for p in range(len(bn.V))]
            
            # step 0: remove all entries from all the tables except for node and its parents
            rowstokeep = [x]
            for z in range(len(bn.V)):
                if bn.V[z] in bn.Vdata[bn.V[x]]["parents"]:
                    rowstokeep.append(z)
            smalldim = len(rowstokeep)
            smallsystem = [[0 for _ in range(smalldim)] for __ in range(smalldim)]
            smallcov = [[0 for _ in range(smalldim)] for __ in range(smalldim)]
            smallsing = [0 for _ in range(smalldim)]
            for index in range(len(rowstokeep)):
                smallsing[index] = singletons[rowstokeep[index]]
                for index2 in range(len(rowstokeep)):
                    smallsystem[index][index2] = system[rowstokeep[index]][rowstokeep[index2]]
                    smallcov[index][index2] = cov[rowstokeep[index]][rowstokeep[index2]]
        
            # step 1: delete and copy row corresponding to node (using [row][column] notation)
            tmparray = [0 for _ in range(smalldim)]
            for y in range(smalldim):
                if (y > 0):
                    for j in range(smalldim):
                        smallsystem[y-1][j] = smallsystem[y][j]
                if (y == 0):
                    for j in range(smalldim):
                        tmparray[j] = smallsystem[y][j]
         
            # step 2: delete column, leaving system with all entries
            # corresponding to parents of node
            for y in range(smalldim):
                if (y > 0):
                    for j in range(smalldim):
                        smallsystem[j][y-1] = smallsystem[j][y]

            # step 3: take entry for node out of singleton array and store it
            bordarray = []
            for y in range(smalldim):
                if (y != 0):
                    bordarray.append(smallsing[y])
                else:
                    tmpentry = smallsing[y]

            # step 4: add border array on borders of system
            for y in range(len(bordarray)):
                smallsystem[smalldim - 1][y] = bordarray[y]
                smallsystem[y][smalldim - 1] = bordarray[y]
            smallsystem[smalldim - 1][smalldim - 1] = 1

            # step 5: construct equality vector (the 'b' of ax = b)
            evector = [0 for _ in range(smalldim)]
            for y in range(smalldim):
                if (y != smalldim - 1):
                    evector[y] = tmparray[y + 1]
                else:
                    evector[y] = tmpentry

            # use numpy to solve
            a = np.array(smallsystem)
            b = np.array(evector)
            solve = list(np.linalg.solve(a, b))
            
            # fill mean_base and mean_scal[] with this data
            bn.Vdata[bn.V[x]]["mean_base"] = solve[smalldim - 1]
            for i in range(smalldim - 1):
                bn.Vdata[bn.V[x]]["mean_scal"][i] = solve[i]

            # add variance
            variance = smallcov[0][0]
            for y in range(1, smalldim):
                for z in range(1, smalldim):
                    variance -= (bn.Vdata[bn.V[x]]["mean_scal"][y-1] * bn.Vdata[bn.V[x]]["mean_scal"][z-1] * smallcov[y][z]) 
            bn.Vdata[bn.V[x]]["variance"] = variance

        # that's all folks
        return bn

    def discrete_constraint_estimatestruct(self, data, pvalparam=0.05, indegree=1):
        '''
        Learn a Bayesian network structure from discrete data given by *data*, using constraint-based approaches. This function first calculates all the independencies and conditional independencies present between variables in the data. To calculate dependencies, it uses the *discrete_condind* method on each pair of variables, conditioned on other sets of variables of size *indegree* or smaller, to generate a chi-squared result and a p-value. If this p-value is less than *pvalparam*, the pair of variables are considered dependent conditioned on the variable set. Once all true dependencies -- pairs of variables that are dependent no matter what they are conditioned by -- are found, the algorithm uses these dependencies to construct a directed acyclic graph. It returns this DAG in the form of a :doc:`GraphSkeleton <graphskeleton>` class. 

        Arguments:
            1. *data* -- An array of dicts containing samples from the network in {vertex: value} format. Example::

                    [
                        {
                            'Grade': 'B',
                            'SAT': 'lowscore',
                            ...
                        },
                        ...
                    ]

            2. *pvalparam* -- (Optional, default is 0.05) The p-value below which to consider something significantly unlikely. A common number used is 0.05. This is passed to *discrete_condind* when it is called.
            3. *indegree* -- (Optional, default is 1) The upper bound on the size of a witness set (see Koller et al. 85). If this is larger than 1, a huge amount of samples in *data* are required to avoid a divide-by-zero error.

        Usage example: this would learn structure from a set of 8000 discrete samples::

            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            from external.libpgm.pgmlearner import PGMLearner
            
            # generate some data to use
            nd = NodeData()
            nd.load("../tests/unittestdict.txt")    # an input file
            skel = GraphSkeleton()
            skel.load("../tests/unittestdict.txt")
            skel.toporder()
            bn = DiscreteBayesianNetwork(skel, nd)
            data = bn.randomsample(8000)

            # instantiate my learner 
            learner = PGMLearner()

            # estimate structure
            result = learner.discrete_constraint_estimatestruct(data)

            # output
            print json.dumps(result.E, indent=2)

        '''
        assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."

        # instantiate array of variables and array of potential dependencies
        variables = list(data[0].keys())
        print(variables)
        ovariables = variables[:]
        dependencies = []
        for x in variables:
            ovariables.remove(x)
            for y in ovariables:
                if (x != y):
                    dependencies.append([x, y])


        # define helper function to find subsets
        def subsets(array):
            result = []
            for i in range(indegree + 1):
                comb = itertools.combinations(array, i)
                for c in comb:
                    result.append(list(c))
            return result

        witnesses = []
        othervariables = variables[:]

        # for each pair of variables X, Y:
        for X in variables:
            othervariables.remove(X)
            for Y in othervariables:

                # consider all sets of witnesses that do not have X or Y in
                # them, and are less than or equal to the size specified by 
                # the "indegree" argument
                for U in subsets(variables):
                    if (X not in U) and (Y not in U) and len(U) <= indegree:
                        
                        # determine conditional independence
                        chi, pv, witness = self.discrete_condind(data, X, Y, U)
                        if pv > pvalparam: 
                            msg = "***%s and %s are found independent (chi = %f, pv = %f) with witness %s***" % (X, Y, chi, pv, U)
                            try:
                                dependencies.remove([X, Y])
                                dependencies.remove([Y, X])
                            except:
                                pass
                            witnesses.append([X, Y, witness])
                            break

        # now that we have found our dependencies, run build PDAG (cf. Koller p. 89) 
        # with the stored set of independencies:
        
        # assemble undirected graph skeleton
        pdag = GraphSkeleton()
        pdag.E = dependencies
        pdag.V = variables
        
        # adjust for immoralities (cf. Koller 86)
        dedges = [x[:] for x in pdag.E]
        for edge in dedges:
            edge.append('u')

        # define helper method "exists_undirected_edge"
        def exists_undirected_edge(one_end, the_other_end):
            for edge in dedges:
                if len(edge) == 3:
                    if (edge[0] == one_end and edge[1] == the_other_end):
                        return True
                    elif (edge[1] == one_end and edge[0] == the_other_end):
                        return True
            return False

        # define helper method "exists_edge"
        def exists_edge(one_end, the_other_end):
            if exists_undirected_edge(one_end, the_other_end):
                return True
            elif [one_end, the_other_end] in dedges:
                return True
            elif [the_other_end, one_end] in dedges: 
                return True
            return False

        for edge1 in reversed(dedges):
            for edge2 in reversed(dedges):
                if (edge1 in dedges) and (edge2 in dedges):
                    if edge1[0] == edge2[1] and not exists_edge(edge1[1], edge2[0]):
                        if (([edge1[1], edge2[0], [edge1[0]]] not in witnesses) and ([edge2[0], edge1[1], [edge1[0]]] not in witnesses)): 
                            dedges.append([edge1[1], edge1[0]])
                            dedges.append([edge2[0], edge2[1]])
                            dedges.remove(edge1)
                            dedges.remove(edge2)
                    elif edge1[1] == edge2[0] and not exists_edge(edge1[0], edge2[1]):
                        if (([edge1[0], edge2[1], [edge1[1]]] not in witnesses) and ([edge2[1], edge1[0], [edge1[1]]] not in witnesses)): 
                            dedges.append([edge1[0], edge1[1]])
                            dedges.append([edge2[1], edge2[0]])
                            dedges.remove(edge1)
                            dedges.remove(edge2)
                    elif edge1[1] == edge2[1] and edge1[0] != edge2[0] and not exists_edge(edge1[0], edge2[0]):
                        if (([edge1[0], edge2[0], [edge1[1]]] not in witnesses) and ([edge2[0], edge1[0], [edge1[1]]] not in witnesses)): 
                            dedges.append([edge1[0], edge1[1]])
                            dedges.append([edge2[0], edge2[1]])
                            dedges.remove(edge1)
                            dedges.remove(edge2)
                    elif edge1[0] == edge2[0] and edge1[1] != edge2[1] and not exists_edge(edge1[1], edge2[1]):
                        if (([edge1[1], edge2[1], [edge1[0]]] not in witnesses) and ([edge2[1], edge1[1], [edge1[0]]] not in witnesses)): 
                            dedges.append([edge1[1], edge1[0]])
                            dedges.append([edge2[1], edge2[0]])
                            dedges.remove(edge1)
                            dedges.remove(edge2)


        # use right hand rules to improve graph until convergence (Koller 89)
        olddedges = []
        while (olddedges != dedges):
            olddedges = [x[:] for x in dedges]
            for edge1 in reversed(dedges):
                for edge2 in reversed(dedges):
                    
                    # rule 1
                    inverted = False
                    check1, check2 = False, True
                    if (edge1[1] == edge2[0] and len(edge1) == 2 and len(edge2) == 3):
                        check1 = True
                    elif (edge1[1] == edge2[1] and len(edge1) == 2 and len(edge2) == 3):
                        check = True
                        inverted = True 
                    for edge3 in dedges:
                        if edge3 != edge1 and ((edge3[0] == edge1[0] and edge3[1]
                            == edge2[1]) or (edge3[1] == edge1[0] and edge3[0]
                            == edge2[1])):
                            check2 = False
                    if check1 == True and check2 == True:
                        if inverted:
                            dedges.append([edge1[1], edge2[0]])
                        else:
                            dedges.append([edge1[1], edge2[1]])
                        dedges.remove(edge2)

                    # rule 2
                    check1, check2 = False, False
                    if (edge1[1] == edge2[0] and len(edge1) == 2 and len(edge2) == 2):
                        check1 = True
                    for edge3 in dedges:
                        if ((edge3[0] == edge1[0] and edge3[1]
                            == edge2[1]) or (edge3[1] == edge1[0] and edge3[0]
                            == edge2[1]) and len(edge3) == 3):
                            check2 = True
                    if check1 == True and check2 == True:
                        if edge3[0] == edge1[0]:
                            dedges.append([edge3[0], edge3[1]])
                        elif edge3[1] == edge1[0]:
                            dedges.append([edge3[1], edge3[0]])
                        dedges.remove(edge3)

                    # rule 3
                    check1, check2 = False, False
                    if len(edge1) == 2 and len(edge2) == 2:
                        if (edge1[1] == edge2[1] and edge1[0] != edge2[0]):
                            check1 = True
                    for v in variables:
                        if (exists_undirected_edge(v, edge1[0]) and
                            exists_undirected_edge(v, edge1[1]) and
                            exists_undirected_edge(v, edge2[0])):
                            check2 = True
                            if check1 == True and check2 == True:
                                dedges.append([v, edge1[1]])
                                for edge3 in dedges:
                                    if (len(edge3) == 3 and ((edge3[0] == v and edge3[1]
                                        == edge1[1]) or (edge3[1] == v and edge3[0] ==
                                        edge1[1]))):
                                        dedges.remove(edge3)
                    

        # return one possible graph skeleton from the pdag class found
        for x in range(len(dedges)):
            if len(dedges[x]) == 3:
                dedges[x] = dedges[x][:2]
        
        pdag.E = dedges
        pdag.toporder()
        return pdag

    def lg_constraint_estimatestruct(self, data, pvalparam=0.05, bins=10, indegree=1):
        '''
        Learn a Bayesian network structure from linear Gaussian data given by *data* using constraint-based approaches. This function works by discretizing the linear Gaussian data into *bins* number of bins, and running the *discrete_constraint_estimatestruct* method on that discrete data with *pvalparam* and *indegree* as arguments. It returns the :doc:`GraphSkeleton <graphskeleton>` instance returned by this function. 

        Arguments:
            1. *data* -- An array of dicts containing samples from the network in {vertex: value} format. Example::

                    [
                        {
                            'Grade': 78.3223,
                            'SAT': 56.33,
                            ...
                        },
                        ...
                    ]

            2. *pvalparam* -- (Optional, default is 0.05) The p-value below which to consider something significantly unlikely. A common number used is 0.05
            3. *bins* -- (Optional, default is 10) The number of bins to discretize the data into. The method is to find the highest and lowest value, divide that interval uniformly into a certain number of bins, and place the data inside. This number must be chosen carefully in light of the number of trials. There must be at least 5 trials in every bin, with more if the indegree is increased.
            4. *indegree* -- (Optional, default is 1) The upper bound on the size of a witness set (see Koller et al. 85). If this is larger than 1, a huge amount of trials are required to avoid a divide-by-zero error.

        The number of bins and indegree must be chosen carefully based on the size and nature of the data set. Too many bins will lead to not enough data per bin, while too few bins will lead to dependencies not getting noticed.

        Usage example: this would learn structure from a set of 8000 linear Gaussian samples::

            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.lgbayesiannetwork import LGBayesianNetwork
            from external.libpgm.pgmlearner import PGMLearner
            
            # generate some data to use
            nd = NodeData()
            nd.load("../tests/unittestdict.txt")    # an input file
            skel = GraphSkeleton()
            skel.load("../tests/unittestdict.txt")
            skel.toporder()
            lgbn = LGBayesianNetwork(skel, nd)
            data = lgbn.randomsample(8000)

            # instantiate my learner 
            learner = PGMLearner()

            # estimate structure
            result = learner.lg_constraint_estimatestruct(data)

            # output
            print json.dumps(result.E, indent=2)

        '''
        assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."
        cdata = copy.deepcopy(data)

        # establish ranges
        ranges = dict()
        for variable in list(cdata[0].keys()):
            ranges[variable] = [float("infinity"), float("infinity") * -1]
        for sample in cdata:
            for var in list(sample.keys()):
                if sample[var] < ranges[var][0]:
                    ranges[var][0] = sample[var]
                if sample[var] > ranges[var][1]:
                    ranges[var][1] = sample[var]

        # discretize cdata set
        bincounts = dict()
        for key in list(cdata[0].keys()):
            bincounts[key] = [0 for _ in range(bins)]
        for sample in cdata:
            for i in range(bins):
                for var in list(sample.keys()):
                    if (sample[var] >= (ranges[var][0] + (ranges[var][1] - ranges[var][0]) * i / float(bins)) and (sample[var] <= (ranges[var][0] + (ranges[var][1] - ranges[var][0]) * (i + 1) / float(bins)))):
                        sample[var] = i 
                        bincounts[var][i] += 1 

        # run discrete_constraint_estimatestruct
        return self.discrete_constraint_estimatestruct(cdata, pvalparam, indegree)


    def discrete_condind(self, data, X, Y, U):
        '''
        Test how independent a variable *X* and a variable *Y* are in a discrete data set given by *data*, where the independence is conditioned on a set of variables given by *U*. This method works by assuming as a null hypothesis that the variables are conditionally independent on *U*, and thus that:

        .. math::

            P(X, Y, U) = P(U) \\cdot P(X|U) \\cdot P(Y|U) 

        It tests the deviance of the data from this null hypothesis, returning the result of a chi-square test and a p-value.

        Arguments:
            1. *data* -- An array of dicts containing samples from the network in {vertex: value} format. Example::

                    [
                        {
                            'Grade': 'B',
                            'SAT': 'lowscore',
                            ...
                        },
                        ...
                    ]
            2. *X* -- A variable whose dependence on Y we are testing given U.
            3. *Y* -- A variable whose dependence on X we are testing given U.
            4. *U* -- A list of variables that are given.

        Returns:
            1. *chi* -- The result of the chi-squared test on the data. This is a
                   measure of the deviance of the actual distribution of X and
                   Y given U from the expected distribution of X and Y given U.
                   Since the null hypothesis is that X and Y are independent 
                   given U, the expected distribution is that :math:`P(X, Y, U) =
                   P(U) P(X | U) P (Y | U)`.
            2. *pval* -- The p-value of the test, meaning the probability of
                    attaining a chi-square result as extreme as or more extreme
                    than the one found, assuming that the null hypothesis is
                    true. (e.g., a p-value of .05 means that if X and Y were 
                    independent given U, the chance of getting a chi-squared
                    result this high or higher are .05)
            3. *U* -- The 'witness' of X and Y's independence. This is the variable
                 that, when it is known, leaves X and Y independent.

        For more information see Koller et al. 790.
        
        '''
        # find possible outcomes and store
        _outcomes = dict()
        for key in list(data[0].keys()):
            _outcomes[key] = [data[0][key]]
        for sample in data:
            for key in list(_outcomes.keys()):
                if _outcomes[key].count(sample[key]) == 0:
                    _outcomes[key].append(sample[key])

        # store number of outcomes for X, Y, and U
        Xnumoutcomes = len(_outcomes[X])
        Ynumoutcomes = len(_outcomes[Y])
        Unumoutcomes = []
        for val in U:
            Unumoutcomes.append(len(_outcomes[val]))

        # calculate P(U) -- the distribution of U
        PU = 1
        
        # define helper function to add a dimension to an array recursively
        def add_dimension_to_array(mdarray, size):
            if isinstance(mdarray, list):
                for h in range(len(mdarray)):
                    mdarray[h] = add_dimension_to_array(mdarray[h], size)
                return mdarray
            else:
                mdarray = [0 for _ in range(size)]
                return mdarray

        # make PU the right size
        for size in Unumoutcomes:
            PU = add_dimension_to_array(PU, size)

        # fill with data
        if (len(U) > 0):
            for sample in data:
                tmp = PU
                for x in range(len(U)-1):
                    Uindex = _outcomes[U[x]].index(sample[U[x]])
                    tmp = tmp[Uindex]
                lastindex = _outcomes[U[-1]].index(sample[U[-1]])
                tmp[lastindex] += 1

        # calculate P(X, U) -- the distribution of X and U
        PXandU = [0 for _ in range(Xnumoutcomes)]
        for size in Unumoutcomes:
            PXandU = add_dimension_to_array(PXandU, size)

        for sample in data:
            Xindex = _outcomes[X].index(sample[X])
            if len(U) > 0: 
                tmp = PXandU[Xindex]
                for x in range(len(U)-1):
                    Uindex = _outcomes[U[x]].index(sample[U[x]])
                    tmp = tmp[Uindex]
                lastindex = _outcomes[U[-1]].index(sample[U[-1]])
                tmp[lastindex] += 1
            else:
                PXandU[Xindex] += 1

        # calculate P(Y, U) -- the distribution of Y and U
        PYandU = [0 for _ in range(Ynumoutcomes)]
        for size in Unumoutcomes:
            PYandU = add_dimension_to_array(PYandU, size)
        for sample in data:
            Yindex = _outcomes[Y].index(sample[Y])
            if len(U) > 0: 
                tmp = PYandU[Yindex]
                for x in range(len(U)-1):
                    Uindex = _outcomes[U[x]].index(sample[U[x]])
                    tmp = tmp[Uindex]
                lastindex = _outcomes[U[-1]].index(sample[U[-1]])
                tmp[lastindex] += 1
            else:
                PYandU[Yindex] += 1

        # assemble P(U)P(X|U)P(Y|U) -- the expected distribution if X and Y are
        # independent given U.
        expected = [[ 0 for _ in range(Ynumoutcomes)] for __ in range(Xnumoutcomes)] 

        # define helper function to multiply the entries of two matrices
        def multiply_entries(matrixa, matrixb):
            matrix1 = copy.deepcopy(matrixa)
            matrix2 = copy.deepcopy(matrixb)
            if isinstance(matrix1, list):
                for h in range(len(matrix1)):
                    matrix1[h] = multiply_entries(matrix1[h], matrix2[h])
                return matrix1
            else:
                return (matrix1 * matrix2)

        # define helper function to divide the entries of two matrices
        def divide_entries(matrixa, matrixb):
            matrix1 = copy.deepcopy(matrixa)
            matrix2 = copy.deepcopy(matrixb)
            if isinstance(matrix1, list):
                for h in range(len(matrix1)):
                    matrix1[h] = divide_entries(matrix1[h], matrix2[h])
                return matrix1
            else:
                return (matrix1 / float(matrix2))

        # combine known graphs to calculate P(U)P(X|U)P(Y|U)
        for x in range(Xnumoutcomes):
            for y in range(Ynumoutcomes):
                product = multiply_entries(PXandU[x], PYandU[y])
                final = divide_entries(product, PU)
                expected[x][y] = final

        # find P(XYU) -- the actual distribution of X, Y, and U -- in sample
        PXYU = [[ 0 for _ in range(Ynumoutcomes)] for __ in range(Xnumoutcomes)] 
        for size in Unumoutcomes:
            PXYU = add_dimension_to_array(PXYU, size)
        
        for sample in data:
            Xindex = _outcomes[X].index(sample[X])
            Yindex = _outcomes[Y].index(sample[Y])
            if len(U) > 0:
                tmp = PXYU[Xindex][Yindex]
                for x in range(len(U)-1):
                    Uindex = _outcomes[U[x]].index(sample[U[x]])
                    tmp = tmp[Uindex]
                lastindex = _outcomes[U[-1]].index(sample[U[-1]])
                tmp[lastindex] += 1
            else:
                PXYU[Xindex][Yindex] += 1 

        # use scipy's chisquare to determine the deviance of the evidence
        a = np.array(expected)
        a = a.flatten()
        b = np.array(PXYU)
        b = b.flatten()

        # delete entries with value 0 (they mess up the chisquare function)
        for i in reversed(range(b.size)):
            if (b[i] == 0):
                if i != 0:
                    a.itemset(i-1, a[i-1]+a[i])
                a = np.delete(a, i)
                b = np.delete(b, i)

        # run chi-squared
        chi, pv = chisquare(a, b)

        # return chi-squared result, p-value for that result, and witness
        return chi, pv, U

    def discrete_estimatebn(self, data, pvalparam=.05, indegree=1):
        '''
        Fully learn a Bayesian network from discrete data given by *data*. This function combines the *discrete_constraint_estimatestruct* method (where it passes in the *pvalparam* and *indegree* arguments) with the *discrete_mle_estimateparams* method. It returns a complete :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` class instance learned from the data.

        Arguments:
            1. *data* -- An array of dicts containing samples from the network in {vertex: value} format. Example::

                    [
                        {
                            'Grade': 'B',
                            'SAT': 'lowscore',
                            ...
                        },
                        ...
                    ]
            2. *pvalparam* -- The p-value below which to consider something significantly unlikely. A common number used is 0.05
            3. *indegree* -- The upper bound on the size of a witness set (see Koller et al. 85). If this is larger than 1, a huge amount of trials are required to avoid a divide-by- zero error.

        '''
        assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."

        # learn graph skeleton
        skel = self.discrete_constraint_estimatestruct(data, pvalparam=pvalparam, indegree=indegree)

        # learn parameters
        bn = self.discrete_mle_estimateparams(skel, data)

        # return
        return bn

    def lg_estimatebn(self, data, pvalparam=.05, bins=10, indegree=1):
        '''
        Fully learn a Bayesian network from linear Gaussian data given by *data*. This function combines the *lg_constraint_estimatestruct* method (where it passes in the *pvalparam*, *bins*, and *indegree* arguments) with the *lg_mle_estimateparams* method. It returns a complete :doc:`LGBayesianNetwork <discretebayesiannetwork>` class instance learned from the data.

        Arguments:
            1. *data* -- An array of dicts containing samples from the network in {vertex: value} format. Example::

                    [
                        {
                            'Grade': 75.23423,
                            'SAT': 873.42342,
                            ...
                        },
                        ...
                    ]
            2. *pvalparam* -- The p-value below which to consider something significantly unlikely. A common number used is 0.05
            3. *indegree* -- The upper bound on the size of a witness set (see Koller et al. 85). If this is larger than 1, a huge amount of trials are required to avoid a divide-by- zero error.

        Usage example: this would learn entire Bayesian networks from sets of 8000 data points::

            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.lgbayesiannetwork import LGBayesianNetwork
            from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            from external.libpgm.pgmlearner import PGMLearner

            # LINEAR GAUSSIAN
            
            # generate some data to use
            nd = NodeData()
            nd.load("../tests/unittestlgdict.txt")    # an input file
            skel = GraphSkeleton()
            skel.load("../tests/unittestdict.txt")
            skel.toporder()
            lgbn = LGBayesianNetwork(skel, nd)
            data = lgbn.randomsample(8000)

            # instantiate my learner 
            learner = PGMLearner()

            # learn bayesian network
            result = learner.lg_estimatebn(data)

            # output
            print json.dumps(result.E, indent=2)
            print json.dumps(result.Vdata, indent=2)

            # DISCRETE

            # generate some data to use
            nd = NodeData()
            nd.load("../tests/unittestdict.txt")    # an input file
            skel = GraphSkeleton()
            skel.load("../tests/unittestdict.txt")
            skel.toporder()
            bn = DiscreteBayesianNetwork(skel, nd)
            data = bn.randomsample(8000)

            # instantiate my learner 
            learner = PGMLearner()

            # learn bayesian network
            result = learner.discrete_estimatebn(data)

            # output
            print json.dumps(result.E, indent=2)
            print json.dumps(result.Vdata, indent=2)

        '''
        assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."

        # learn graph skeleton
        skel = self.lg_constraint_estimatestruct(data, pvalparam=pvalparam, bins=bins, indegree=indegree)

        # learn parameters
        bn = self.lg_mle_estimateparams(skel, data)

        # return
        return bn


