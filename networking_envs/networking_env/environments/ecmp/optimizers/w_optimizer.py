import numpy as np
import networkx as nx
from copy import deepcopy
from networking_env.environments.consts import HistoryConsts
from networking_env.environments.ecmp.optimizers.optimizer import Optimizer

class WOptimizer(Optimizer):

    def __init__(self, adj, c_vec, props=None):
        super(WOptimizer,self).__init__(adj,c_vec,props)
        
        self._max_iters = props.optimization_steps if props is not None else 500
        self._nx_graph = False if props is None else props.nx_graph
        self._from_routing = True if props is None else props.from_routing

    def direct_q_value(self):
        self._compute_q_val = False
    
    def _init(self, adj, c_vec):
        self._compute_q_val = True
        self._mask = np.ones((self._num_nodes,self._num_nodes), dtype=np.float32)-np.eye(self._num_nodes, dtype=np.float32)
        self._eye_mask = np.eye(self._num_nodes, dtype=np.float32)
        self._eye_masks = [np.expand_dims(self._mask[:,i],1) for i in range(self._num_nodes)]
        self._mask = np.transpose(self._mask)
        self._q_zero_one = self._a_out.copy()
        
        self._nonzero_a_out = np.nonzero(self._a_out)
        self._a_out_multi = np.tile(self._a_out,(1,self._num_nodes))
        
        self._routing_demand_sum = {k: np.ones((k, 1)) for k in range(1,self._num_nodes)}
        
        self._demand_for_dst = {}
        for dst in range(self._num_nodes):
            demand = np.eye( self._num_nodes )
            demand = np.delete(demand, dst, 0)
            demand = np.reshape(demand, ( self._num_nodes-1, self._num_nodes,1) )
#             
            self._demand_for_dst[dst] = demand
        
        self._eye_nodes = np.eye(self._num_nodes)

    def _softmin(self, v, axis=1, alpha=HistoryConsts.SOFTMIN_ALPHA):
        # this is semi true, we need to take into account the in degree of things
        # as the softmax is vector dependet
        # if we assume v is a matrix this will help, and now we run softmin
        # across the per vector direction
        
        exp_val = np.exp(alpha*v)
        exp_val[v==0]=0
        exp_val[np.logical_and(v!=0,exp_val==0) ] = HistoryConsts.EPSILON
        
        exp_val = np.transpose(exp_val)/np.sum(exp_val, axis=1)
        
        return np.transpose(exp_val)  #sum over edges  
    
    def _softmax(self, v, axis=1, alpha=HistoryConsts.SOFTMAX_ALPHA):
        # this is semi true, we need to take into account the in degree of things
        # as the softmax is vector dependet
        # if we assume v is a matrix this will help, and now we run softmin
        # across the per vector direction
        
        exp_val = np.exp(alpha*v)
        exp_val[v==0]=0
        exp_val[np.logical_and(v!=0,exp_val==0) ] = HistoryConsts.EPSILON
        
        exp_val = np.transpose(exp_val)/np.sum(exp_val, axis=1)
        
        return np.transpose(exp_val)  #sum over edges  

    def _actual_min(self, v):
        cpy = v.copy()
        cpy[cpy==0] = HistoryConsts.INFTY
        cpy[(cpy-np.amin(cpy,axis=1)[:,None]) != 0 ] = 0
        cpy[cpy!=0 ] = 1
        cpy /= np.sum(cpy,axis=1)[:,None]
        return cpy
    
    def _bellman_ford(self, w, p_prev, tmp):
        p_prev = np.reshape(p_prev,[-1])
        prev_prev =  np.zeros_like(p_prev)
        cur_iter=0
        while np.abs(np.sum(p_prev-prev_prev)) > HistoryConsts.EPSILON:
            tmp_ = p_prev
            p_prev = np.minimum(p_prev, np.amin(p_prev+tmp, axis=1) )
            prev_prev = tmp_
            if cur_iter==self._max_iters:
                break
            cur_iter+=1
        return p_prev #TODO: dst val is infty here

    def _get_edge_cost_v2(self, w, a_out, a_in, cost_adj, adj, tmp):
        cost_to_dst1 = cost_adj*adj+tmp
        cost_to_dst2 = np.reshape(cost_to_dst1, [-1])
        cost_to_dst3 =  cost_to_dst2[cost_to_dst2!=0]
        return cost_to_dst3 * a_out 
    
    def _get_s_at_dst(self, q, a_in, a_out, demand, mask, dst=None):
        '''
        input:
            v: the node with demands
            demand: the demand of node v
            q: is the softmax cost vector matrix (dim = |V|x|E|)
            e_in: is the relationship matrix, a_ie=1 iff e \in In(i) (dim = |V|x|E|)
        output:
            s: is the flow input vector (dim=|E|)
        '''
    
        # loop magic goes here, basically converts the for loop into matrix operations
        def get_new_val(prev_val, mul_val):
            res = prev_val + a_in @ ( np.transpose(q * a_out) @ mul_val) 
            return res
        
        prev = get_new_val(demand, demand)
        prev_prev = demand
        cur_iter = 0
        demand_sum = np.sum(demand)
        def comp(p):
            if dst is None:
                return np.sum(prev*(1-mask))/demand_sum
            else:
                return prev[dst]/demand_sum
        
        while comp(prev) < HistoryConsts.PERC_DEMAND :
            res_diff = (prev - prev_prev)*mask
            tmp = get_new_val(prev, res_diff)
            prev_prev = prev
            prev = tmp
            if cur_iter==self._max_iters:
                break
            cur_iter+=1
            
        final_s_value = prev
        edge_congestion = np.sum( np.transpose(q) @ (final_s_value*mask) ,axis=1 )
        
        return edge_congestion, prev  


    def _get_s_change(self, q, a_in, a_out, demand, mask):
        '''
        input:
            v: the node with demands
            demand: the demand of node v
            q: is the softmax cost vector matrix (dim = |V|x|E|)
            e_in: is the relationship matrix, a_ie=1 iff e \in In(i) (dim = |V|x|E|)
        output:
            s: is the flow input vector (dim=|E|)
        '''
    
        # loop magic goes here, basically converts the for loop into matrix operations
        def get_new_val(prev_val, mul_val):
            res = prev_val + a_in @ ( np.transpose(q * a_out) @ mul_val) 
            return res
        
        prev = get_new_val(demand, demand)
        prev_prev = demand
        cur_iter = 0
        # continue while there is still some change in the network
        while np.sum(prev-prev_prev) > HistoryConsts.EPSILON :
            
            res_diff = (prev - prev_prev)*mask
            tmp = get_new_val(prev, res_diff)
            prev_prev = prev
            prev = tmp
            if cur_iter==self._max_iters:
                break
            cur_iter+=1
            
        final_s_value = prev
        edge_congestion = np.sum( np.transpose(q) @ (final_s_value*mask) ,axis=1 )
        return edge_congestion, prev

    def _get_routing_splitting_ratios(self, q, a_in, a_out, demand, mask, dst=None):
        '''
        input:
            v: the node with demands
            demand: the demand of node v
            q: is the softmax cost vector matrix (dim = |V|x|E|)
            e_in: is the relationship matrix, a_ie=1 iff e \in In(i) (dim = |V|x|E|)
        output:
            s: is the flow input vector (dim=|E|)
        '''
        def comp(vec, dst):
            res = (vec*(1-mask))[:,dst,:]/routing_demand_sum
            return np.any(res < HistoryConsts.PERC_DEMAND)
        
        # loop magic goes here, basically converts the for loop into matrix operations
        # right hand side ( np.transpose(q * a_out) @ traffic_vec ) :
        # send all the traffic in traffic_vec using the splitting ratios given by q
        # left hand side (a_in @ ): 
        # get all of the incoming traffic into the nodes 
        def get_new_val( traffic_vec):
            return a_in @ ( ( np.transpose(q * a_out) @ traffic_vec ) )
        
        def flow_on_edges( traffic_vec ):
            return np.transpose(q * a_out) @ traffic_vec
        
        # this is to create fake demands of one unit of traffic
        nodes_with_demands = np.nonzero(demand)[0]
        demand_nonzero = demand[nodes_with_demands]
        routing_demand_sum = self._routing_demand_sum[len(nodes_with_demands)]
        
        # compute how much data we have from each node
        if nodes_with_demands is None:
            demand = deepcopy(self._demand_for_dst[dst])
            num_with_demand = self._num_nodes-1
        else:
            demand = deepcopy(self._eye_nodes[nodes_with_demands])
            demand = np.expand_dims(demand,-1)
            num_with_demand = len(nodes_with_demands)
        
        # initially we have just our demand
        aggregate_traffic_at_node = demand
        cur_iter = 0
        leftover = demand
        
        traffic_on_edges = np.zeros_like(flow_on_edges(demand))
        # send all new traffic until we've sent everything to destination
        while comp(aggregate_traffic_at_node,dst) :
            # how much new data do we have to send, that did not reach the destination
            leftover = np.multiply(leftover, mask)
            
            # route that traffic to our neighbors
            flow_new = flow_on_edges(leftover)
            traffic_on_edges += flow_new 
            leftover = a_in@flow_new
            aggregate_traffic_at_node += leftover

            # allow for MAX_ITER until declared as converged
            if cur_iter==self._max_iters:
                break
            cur_iter+=1
        

        routing = np.reshape(traffic_on_edges, (num_with_demand,self._num_edges))
        
        congestion = np.sum(demand_nonzero*routing,0)
        return congestion 

    def _set_cost_to_dsts(self, w):
        tmp = (w*self._a_out) @ np.transpose(self._a_in)
        tmp[tmp==0]=HistoryConsts.INFTY 
        return tmp*self._zero_diagonal
    
    def _get_w_cost(self, w, demand_split):
        if self._compute_q_val:
            one_hop_cost = (w*self._a_out) @ np.transpose(self._a_in)
            initial_costs = self._set_cost_to_dsts(w)
        else:
            w_split = np.split(w, self._num_nodes)
        
        def func_with_q(*x):
            demand, mask, first_dist = x
            demand = np.expand_dims(demand, 1)
            mask = np.expand_dims(mask, 1)
            first_dist *= np.reshape(mask,[-1]) 
            
            cost_adj = self._bellman_ford(w, first_dist, initial_costs)
            edge_cost = self._get_edge_cost_v2(w, self._a_out, self._a_in, cost_adj,self._adj, one_hop_cost)
            
            q_val = self._softmin(edge_cost)
            cong = self._get_s_at_dst(q_val, self._a_in, self._a_out, demand, mask)
            
            return np.reshape(cong,[-1]) 
        
        def func_without_q(*x):
            demand, mask, q_val_base = x
            demand = np.expand_dims(demand, 1)
            mask = np.expand_dims(mask, 1)
            # need to parse Q, as this is V^3
            q_val = self._softmax(self._a_out*q_val_base)
            cong = self._get_s_at_dst(q_val, self._a_in, self._a_out, demand, mask)
            return np.reshape(cong,[-1]) 
        
        res = np.zeros_like(self._c, dtype=np.float32)
        for i in range(len(demand_split)):
            if self._compute_q_val:
                res_ = func_with_q(demand_split[:,i], self._mask[:,i], initial_costs[:,i])
            else:
                # in this case the weight vector is actually the splitting ratios matrix
                res_ = func_without_q(demand_split[:,i], self._mask[:,i], w_split[i])
                
            res += res_
        congestion = res/self._c 
        cost = np.max( congestion  )
        return cost, res
    
    def _get_w_g_cost(self, w, demand_split):
        if self._compute_q_val:
            g_mat = (w*self._a_out) @ np.transpose(self._a_in)
            cost_all_adj=nx.shortest_path_length(nx.from_numpy_matrix(g_mat,create_using=nx.DiGraph()), weight='weight')
        else:
            w_split = np.split(w, self._num_nodes)
        
        def func_with_q(*x):
            dst = x[0]
            demand = np.expand_dims(demand_split[:,dst], 1)
            if np.sum(demand) == 0:
                return 
            cost_adj=[cost_all_adj[i][dst] for i in range(self._num_nodes)]
            edge_cost = self._get_edge_cost_v2(w, self._a_out, self._a_in, cost_adj,
                                               self._adj, g_mat)
            
            
            q_val = self._softmin(edge_cost)
            cong, _ = self._get_s_at_dst(q_val, self._a_in, self._a_out, demand, self._eye_masks[dst])
            return np.reshape(cong,[-1]) 
        
        def func_without_q(*x):
            dst = x[0]
            q_val_base = w_split[0]
            demand = np.expand_dims(demand_split[:,dst], 1)
            # need to parse Q, as this is V^3
            q_val = self._softmax(self._a_out*q_val_base)
            cong, _ = self._get_s_at_dst(q_val, self._a_in, self._a_out, demand, self._eye_masks[dst])
            return np.reshape(cong,[-1])

        res = np.zeros_like(self._c, dtype=np.float32)
        q_comp_func = func_with_q if self._compute_q_val else func_without_q
        for i in range(len(demand_split)):
            iter_res = q_comp_func(i)
            if iter_res is not None:
                res += iter_res  
            
        congestion = res/self._c 
        cost = np.max( congestion  )
        return cost, res
    
    def _get_routing_from_w(self, w, demand_split):
        if self._compute_q_val:
            g_mat = (w*self._a_out) @ np.transpose(self._a_in)
            cost_all_adj = nx.shortest_path_length(nx.from_numpy_matrix(g_mat,create_using=nx.DiGraph()), weight='weight')
            cost_all_adj = dict(cost_all_adj)
        else:
            w_split = np.split(w, self._num_nodes)

        def func_with_q(*x):
            dst = x[0]
            demand = np.expand_dims(demand_split[:, dst], 1)
            if np.sum(demand) == 0:
                # if we have no demand towards a destination skip
                # all computation
                return np.zeros((self._num_edges,))
            
            cost_adj = [cost_all_adj[i][dst] for i in range(self._num_nodes)]
            edge_cost = self._get_edge_cost_v2(w, self._a_out, self._a_in, cost_adj,
                                               self._adj, g_mat)
            
            
            q_val = self._softmin(edge_cost)
            cong = self._get_routing_splitting_ratios(q_val, self._a_in, self._a_out, demand, self._eye_masks[dst], dst)
            return np.reshape(cong,[-1]) 
        
        def func_without_q(*x):
            dst = x[0]
            q_val_base = w_split[0]
            demand = np.expand_dims(demand_split[:,dst], 1)
            # need to parse Q, as this is V^3
            q_val = self._softmax(self._a_out*q_val_base)
            cong = self._get_routing_splitting_ratios(q_val, self._a_in, self._a_out, demand, self._eye_masks[dst])
            return np.reshape(cong,[-1])

        res = np.zeros_like(self._c, dtype=np.float32)
        q_comp_func = func_with_q if self._compute_q_val else func_without_q
        for i in range(len(demand_split)):
            res += q_comp_func(i) 
        congestion = res/self._c 
        
        cost = np.max( congestion  )
        return cost, res
    
    def step_w_first(self, w, tm, cost_only=True):
        return self.step(tm,w,cost_only)
    
    def step(self, w, tm, cost_only=True):
        if self._from_routing:
            cost, congestion = self._get_routing_from_w(w,tm)
        elif not self._nx_graph: 
            cost, congestion = self._get_w_g_cost(w, tm)
        else:
            cost, congestion = self._get_w_cost(w, tm)
        if cost_only:
            return cost
        else:
            return cost, congestion
