import sys 
import os
# os.environ["CUDA_VISIBLE_DEVICES"]= str(2)
import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.stats as sts
import pandas as pd 
import time 

tf.logging.set_verbosity(tf.logging.FATAL)
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

np.random.seed(4)
import argparse


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

# build a config 
arg_lists = []
res_arg = add_argument_group('prediction')
res_arg.add_argument('--task', type=str, default='pred', help='')
res_arg.add_argument('--if_batch_nomalizing', type=str2bool, default='True', help='')
res_arg.add_argument('--use_momentun', type=str2bool, default='True', help='otherwise it uses adam')
res_arg.add_argument('--use_sigmoid', type=str2bool, default='True', help='otherwise it uses relu activation')
res_arg.add_argument('--dis', type=str, default='normal', help='')
res_arg.add_argument('--l2lambda', type=float, default=0.01, help='')
res_arg.add_argument('--lr0', type=float, default=0.01, help='')
res_arg.add_argument('--use_dropout', type=str2bool, default='False', help='')
res_arg.add_argument('--clusters', type=int, default=100, help='')
res_arg.add_argument('--maxiter', type=int, default=2100000, help='')
res_arg.add_argument('--display', type=int, default=20000, help='')
res_arg.add_argument('--batch_size', type=int, default=128, help='')
res_arg.add_argument('--decay_step', type=int, default=10000, help='')
res_arg.add_argument('--decay_rate', type=float, default=0.98, help='')
res_arg.add_argument('--decay_rate_stair', type=float, default=0.98, help='')
res_arg.add_argument('--cp', type=float, default=10, help='')
res_arg.add_argument('--ch', type=float, default=1, help='')
res_arg.add_argument('--num_of_period', type=int, default=11, help='the number of input periods (defualt is 11)')
res_arg.add_argument('--eager_mode', type=str2bool, default='False', help='')
res_arg.add_argument('--nodes', type=list, default=[350, 150], help='')
res_arg.add_argument('--num_outputs', type=int, default=11, help='')
res_arg.add_argument('--larry_test', type=str2bool, default='False', help='')

res_arg.add_argument('--sim_period', type=int, default=11, help='')
res_arg.add_argument('--if_print_eil_solutions', type=str2bool, default='True', help='')
res_arg.add_argument('--if_print_knn_solutions', type=str2bool, default='True', help='') 
res_arg.add_argument('--if_simulation_data', type=str2bool, default='True', help='') 
res_arg.add_argument('--if_print_rQ', type=str2bool, default='False', help='')
res_arg.add_argument('--if_print_rQ_test_set', type=str2bool, default='False', help='')
res_arg.add_argument('--if_print_knn_final_cost', type=str2bool, default='False', help='')
res_arg.add_argument('--use_current_trained_network', type=str2bool, default='False', help='If True, it does not reset and load any network in run_simulator()')

def update_checkpoint(str_num, addr):
    ''' This function removes the address in the saved checkpoint file'''
    import re
    addr = os.path.join(addr, 'checkpoint')    
    fille = open(addr, "r")
    dis = ['beta', 'lognormal', 'normal', 'exponential', 'uniform']
    cls = [200, 100, 10, 1]
    out = ''
    no_enter=False
    for line in fille:
        word_list = line.split(' ')
        print word_list
        for word in word_list:
            print word

            if '\n' in word:
                word=re.sub('\n','', word)

            word = re.sub('rq_runner_code/saved_networks/', '', word)
            for d in dis:
                if d in word:
                    for c in cls:
                        if str(c)+'/' in word:
                            word = re.sub(d+'/'+str(c)+'/', '', word)
                            break
                    break

            for c in reversed(range(100)):
                if str(c)+'/' in word:
                    word = re.sub(str(c)+'/', '', word)
                    break
            out += word
        out += '\n'
    print out
    fille.close()

    fille = open(addr, "w+")
    fille.write(out)
    fille.close()

class rq(object):
    def __init__(self, clusters, real_cluster, dis, config):
        self.config = config
        self.clusters = clusters
        self.dis = dis
        self.real_cluster = real_cluster
        self.if_print_eil_solutions = True
        self.if_print_knn_solutions = True
        self.if_simulation_data = True
        self.if_print_rQ = False
        self.if_print_rQ_test_set = False
        
    def get_data(self):            
        if self.config.if_simulation_data:
            dirname = os.path.abspath('data/')
            if self.config.larry_test:                
                # get data
                train_mat_x = np.load(os.path.join(dirname,'rq_larry_x_train.npy'))
                train_mat_y = np.load(os.path.join(dirname,'rq_larry_y_train.npy'))
                test_mat_x = np.load(os.path.join(dirname,'rq_larry_x_test.npy'))
                test_mat_y = np.load(os.path.join(dirname,'rq_larry_y_test.npy'))
                index_mat_train = np.load(os.path.join(dirname,'rq_larry_ind_train.npy'))
                index_mat_test = np.load(os.path.join(dirname,'rq_larry_ind_test.npy'))
                self.ind_train = index_mat_train
            else:
                dirname = os.path.join(dirname,self.dis)
                # get data
                train_mat_x = sio.loadmat(os.path.join(dirname,'TrainX-nw-10000-'+str(self.real_cluster)+'-class.mat'))
                train_mat_y = sio.loadmat(os.path.join(dirname,'TrainY-nw-10000-'+str(self.real_cluster)+'-class.mat'))
                test_mat_x = sio.loadmat(os.path.join(dirname,'TestX-nw-10000-'+str(self.real_cluster)+'-class.mat'))
                test_mat_y = sio.loadmat(os.path.join(dirname,'TestY-nw-10000-'+str(self.real_cluster)+'-class.mat'))
                index_mat_train = sio.loadmat(os.path.join(dirname,'IndexX-nw-10000-'+str(self.real_cluster)+'-class.mat'))
                index_mat_test = sio.loadmat(os.path.join(dirname,'IndexY-nw-10000-'+str(self.real_cluster)+'-class.mat'))

                train_mat_x = train_mat_x['trainX']
                train_mat_y = train_mat_y['trainY']
                test_mat_x = test_mat_x['testX']
                test_mat_y = test_mat_y['testY']
                self.ind_train = index_mat_train['IndexX']
                index_mat_test = index_mat_test['IndexY']

            test_limit = 1+99
            if self.clusters != 1:            
                self.train_x = train_mat_x[:,0,:]
                print "self.train_x", np.shape(self.train_x)
                self.test_x = test_mat_x[0:test_limit*2500,0,:]            
            else:
                self.train_x = train_mat_x[:,:]
                self.test_x = test_mat_x[0:test_limit*2500,:]

            self.train_y = train_mat_y[:,0]
            self.test_y = test_mat_y[0:test_limit*2500,0]

            # get validation data
            self.valid_x = self.test_x[0:1*2500,:]
            self.valid_y = np.squeeze(test_mat_y[0:1*2500])
            self.ind_valid = index_mat_test[0:1*2500,:]
            self.test_x = self.test_x[1*2500:test_limit*2500,:]
            print "self.test_x", np.shape(self.test_x)
            self.test_y = np.squeeze(test_mat_y[1*2500:test_limit*2500])
            self.ind_test = index_mat_test[1*2500:test_limit*2500,:]


            if self.clusters == 1:
                self.NoInputs = 1
            elif self.clusters == 10 or self.clusters == 100 or self.clusters == 103:
                self.NoInputs = 31
            elif self.clusters == 203 or self.clusters == 200:
                self.NoInputs = 36 
        else:
            test_mat = np.genfromtxt('data/basket_test_data_w_mu_sigma.csv' ,dtype=float, delimiter=',',skip_header=1)
            train_mat = np.genfromtxt('data/basket_train_data_w_mu_sigma.csv' ,dtype=float, delimiter=',',skip_header=1)

            test_binary = np.genfromtxt('data/Basket_test_data_binary.csv' ,dtype=float, delimiter=',',skip_header=0)
            train_binary = np.genfromtxt('data/Basket_train_data_binary.csv' ,dtype=float, delimiter=',',skip_header=0)

            train_mat_ind = np.genfromtxt('data/train_ind.csv' , delimiter=',').astype(int)
            test_mat_ind = np.genfromtxt('data/test_ind.csv' , delimiter=',').astype(int)

            ind_train = np.expand_dims(train_mat_ind,1)
            self.ind_train = ind_train[:9000,:]
            self.ind_test = np.expand_dims(test_mat_ind,1)
            self.ind_valid = ind_train[9000:,:]            

            # cluster number, day of week, month of year, department id, demand, mean of demand for training cluster, 
            # std of demand for training cluster

            self.train_x = np.squeeze(np.array(train_binary[:9000,:43]))
            self.test_x = np.squeeze(np.array(test_binary[:,:43]))
            self.valid_x = np.squeeze(np.array(train_binary[9000:,:43]))
            self.train_y = np.squeeze(np.array(train_binary[:9000,43]))
            self.test_y = np.squeeze(np.array(test_binary[:,43]))
            self.valid_y = np.squeeze(np.array(train_binary[9000:,43]))

        train_size=len(self.train_y)
        test_size=len(self.test_y)
        valid_size=len(self.valid_y)

        self.train_size = train_size
        self.test_size = test_size
        self.valid_size = valid_size
        # define the required cost coefficients of the model.
        self.zeros_tr =np.zeros((train_size,1))
        self.zeros_te =np.zeros((test_size,1))
        self.zeros_val =np.zeros((valid_size,1))

        self.cp = self.config.cp
        self.ch = self.config.ch

        self.shrtg_cost_tr = self.cp*np.ones((train_size,1))
        self.hld_cost_tr =  self.ch*np.ones((train_size,1))
        self.shrtg_cost_te = self.cp*np.ones((test_size,1))
        self.hld_cost_te =  self.ch*np.ones((test_size,1))
        self.shrtg_cost_val = self.cp*np.ones((valid_size,1))
        self.hld_cost_val =  self.ch*np.ones((valid_size,1))
    
    # get mu and sigma of each cluster. We use them later to obtain EIL cost 
    def get_mu_sigma(self):
        if self.clusters == 1:
            nn = 1
            self.mu = np.zeros(nn)
            self.sigma = np.zeros(nn)

            yy = pd.DataFrame(self.train_y)
            xx = pd.DataFrame(self.train_x)

            for i in range(nn):
                self.mu[i] = np.mean(yy.values)
                self.sigma[i] = np.std(yy.values)
            #
            self.il_rep = 0
            self.input_dim = len(self.train_x[0]) + self.il_rep
        else:
            nn = int(np.amax([np.amax(self.ind_train[:,0]), np.amax(self.ind_valid[:,0]),
                              np.amax(self.ind_test[:,0])])) + 1
            a,b = np.unique(self.ind_train[:,0], return_index=True)
            self.mu = np.zeros(nn)
            self.sigma = np.zeros(nn)

            yy = pd.DataFrame(self.train_y)
            xx = pd.DataFrame(self.train_x)
            ii = pd.DataFrame(self.ind_train)

            # loop over the number of clusters and get mu and sigma of each 
            # cluster and save them. 
            for i in range(nn):
                if len(yy[self.ind_train[:,0]==i]) == 0:
                    self.mu[i] = np.mean(yy.values)
                    self.sigma[i] = np.std(yy.values)
                else:                
                    self.mu[i] = np.mean(yy[self.ind_train[:,0]==i].values)
                    self.sigma[i] = np.std(yy[self.ind_train[:,0]==i].values)
            #
            self.il_rep = 0
            self.input_dim = len(self.train_x[0]) + self.il_rep

    def set_dnn_settings(self):
        # if use newsvendor model, set it True, for (s,S) model set it False
        # sS=False
        self.rQ=True
        self.EIL=True
        self.aprx=False
        # if it is ture, it uses relu activation function, otherwise uses sigmoid 
        self.ifRelu = True

        config_tf = tf.ConfigProto()
        # config_tf.gpu_options.per_process_gpu_memory_fraction = 0.1
        config_tf.gpu_options.allow_growth = True
        config_tf.intra_op_parallelism_threads = 1
        self.sess = tf.InteractiveSession(config=config_tf)
        
        cur_dir=os.path.realpath("./saved_networks")
        cur_dir=os.path.join(cur_dir, self.dis)
        self.model_dir=os.path.join(cur_dir, str(self.clusters))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


        self.config.maxiter = 45001
        self.config.display = 7500
        self.config.decay_rate = 0.0005
        self.config.decay_rate_stair = 0.99
        self.config.rl0 = 0.005
        self.power = 0.75
        self.config.l2lambda = 0.009
        self.init_momentum = 0.9
        self.config.decay_step = 15000

        self.run_number = 0

        self.config.batch_size = 128
        self.var = 2.0/44
        self.input_dim = len(self.train_x[0]) + self.il_rep

        if self.rQ:
            self.nodes = [self.input_dim, 90, 150, 56,2]
        else:
            self.nodes = [self.input_dim, 90, 150, 56,1]

        self.NoHiLay = len(self.nodes) - 2


        self.shrtg_cost_tr = self.cp*np.ones((self.train_size,1))
        self.hld_cost_tr = self.ch*np.ones((self.train_size,1))
        self.order_cost_tr = self.K*np.ones((self.train_size,1))
        self.lambdaa_tr = self.lambdaa*np.ones((self.train_size,1))
        self.l_tr = np.ones((self.train_size,1))
        self.zeros_tr = np.zeros((self.train_size,1))
        self.shrtg_cost_te = self.cp*np.ones((self.test_size,1))
        self.hld_cost_te = self.ch*np.ones((self.test_size,1))
        self.order_cost_te = self.K*np.ones((self.test_size,1))
        self.lambdaa_te  = self.lambdaa*np.ones((self.test_size,1))
        self.l_te = np.ones((self.test_size,1))
        self.zeros_te =np.zeros((self.test_size,1))
        self.shrtg_cost_val = self.cp*np.ones((self.valid_size,1))
        self.hld_cost_val = self.ch*np.ones((self.valid_size,1))
        self.order_cost_val = self.K*np.ones((self.valid_size,1))
        self.lambdaa_val  = self.lambdaa*np.ones((self.valid_size,1))
        self.l_val = np.ones((self.valid_size,1))
        self.zeros_val = np.zeros((self.valid_size,1))

        self.loss_type = 'L2'

        for c,i in enumerate(self.ind_test[:,0]):
            if self.clusters == 1:
                i = 0
            self.l_te[c,0] = self.l[int(i)]

        for c,i in enumerate(self.ind_train[:,0]):
            if self.clusters == 1:
                i = 0            
            self.l_tr[c,0] = self.l[int(i)]

        for c,i in enumerate(self.ind_valid[:,0]):
            if self.clusters == 1:
                i = 0            
            self.l_val[c,0] = self.l[int(i)]

    # get K, lambda, and L (lead time demand)
    def set_rq_settings(self):
        self.K = 20

        self.lambdaa = 1200
        if self.clusters == 1:
            self.l = self.mu/self.lambdaa 
        else:
            # note that self.lambdaa is a scallar so that we do not need np.divide 
            self.l = self.mu/self.lambdaa
          
    # get approximated solution of (r,Q) policy, since approximate the distribution by normal 
    def get_EIL_solution(self):
        # get approximated solution of (r,Q) policy, since approximate the distribution by normal 
        #
        # It uses the approximated mu and sigma, obtained in self.get_mu_sigma
        # We assume *know* lambda. This is a much more reasonable assumption under the new parameters 
        # than under the old ones. Basically it means that we know the average demand per *year*, 
        # but the actual demand in any given lead time (2 weeks) depends on the features. 
        # Under the old parameters, we were saying we know the average demand per *day* but not the 
        # average demand over *5 days*, which does not make sense.

        epsilon = 0.1
        big_r = 10000
        if self.clusters == 1:
            self.Q_new = np.sqrt(2*self.K*self.lambdaa/self.ch)*np.ones(self.clusters)
            self.r_new = big_r*np.ones(self.clusters)
        else:
            self.Q_new = np.squeeze(np.sqrt(2*self.K*self.lambdaa/self.ch)*np.ones((1,self.clusters)))
            self.r_new = np.squeeze(big_r*np.ones((1,self.clusters)))
        # print '(r,Q) is: ', r_new, Q_new
        for cls in range(self.clusters):
            notStop = True
            i = 0    
            while notStop:
                i += 1
        #         print "iteration ", i
                # reset the value of r,Q
                self.Q_old = self.Q_new[cls]
                self.r_old = self.r_new[cls]
                # update r
                self.z = (self.Q_old*self.ch)/(self.cp*self.lambdaa)
                if self.z >= 0 and self.z <= 1:
                    if self.sigma[cls] < 1e-8:
                        self.sigma[cls] = 1e-8                    
                    self.r_new[cls] = sts.norm.isf(self.z, self.mu[cls],self.sigma[cls]) # isf works with 1-cdf 
                elif self.z > 1:
                    self.r_new[cls] = -big_r
                # update Q
                self.z = (self.r_new[cls]-self.mu[cls])/self.sigma[cls]
                self.zetta = sts.norm.pdf(self.z) - self.z*(1-sts.norm.cdf(self.z))
                self.nr = self.zetta*self.sigma[cls]
                self.Q_new[cls] = np.sqrt((2*self.lambdaa*(self.K+self.cp*self.nr))/self.ch)
        #         print 'z, zetta, (r,Q) is: ',  z , zetta, r_new[cls], Q_new[cls]    
                # check if we should stop 
                if np.abs(self.Q_new[cls]-self.Q_old) < epsilon:
                    if np.abs(self.r_new[cls]-self.r_old) < epsilon:
                        notStop = False
            if self.if_print_eil_solutions:
                print cls, '( %0.2f' %self.r_new[cls] ,', %0.2f' %self.Q_new[cls], ')', 'g(r,Q) is: %0.2f' %(
                self.ch*(self.r_new[cls] - self.lambdaa*self.l[cls] + self.Q_new[cls]/2) +
                self.K*self.lambdaa/self.Q_new[cls] + self.cp*self.lambdaa*self.nr/self.Q_new[cls])
                , i, 'iterations'
            
    # It gets EIL cost based on given self.r_new and self.Q_new, obtained by 
    # self.get_EIL_solution(). It goes over all demand (which are self.train_y, 
    # self.valid_y, self.test_y) 
    def get_eil_cost(self, demand, ind):
        # r_new_ and Q_new_ are the list with "clusters" members 
        if self.clusters == 1:
            cost = 0
            for d in demand:
                approximate_nr = max(d-self.r_new,0)
                cost += self.ch*(self.r_new - self.lambdaa*self.l[cls] + self.Q_new/2) + \
                self.K*self.lambdaa/self.Q_new + self.cp*self.lambdaa*approximate_nr/self.Q_new
        else:        
            cost = 0
            for i, d in enumerate(demand):
                r_new = self.r_new[int(ind[i][0])]
                Q_new = self.Q_new[int(ind[i][0])]
                approximate_nr = max(d-r_new,0)
                cost += self.ch*(r_new - self.lambdaa*self.l[int(ind[i][0])] + Q_new/2) + \
                self.K*self.lambdaa/Q_new + self.cp*self.lambdaa*approximate_nr/Q_new
        return cost 

    def print_EIL_costs(self):
        '''print EIL cost for all avilable demand data'''
        # get optimal solutions 
        print 'optimal solutions'
        cost_val = self.get_eil_cost(self.valid_y, self.ind_valid)
        cost_tr = self.get_eil_cost(self.train_y, self.ind_train)
        print '\t \t optimal' , '\t', 'DNN'
        print 'train \t %0.1f' %cost_tr
        print 'valid \t %0.1f' %cost_val

    def get_aprx_solution(self):
    '''get approximated solution of (r,Q) policy, since approximate the distribution by normal 

         It uses the approximated mu and sigma, obtained in self.get_mu_sigma
         We assume *know* lambda. This is a much more reasonable assumption under the new parameters 
         than under the old ones. Basically it means that we know the average demand per *year*, 
         but the actual demand in any given lead time (2 weeks) depends on the features. 
         Under the old parameters, we were saying we know the average demand per *day* but not the 
         average demand over *5 days*, which doesn’t make sense.'''

        def solve_nr(r):
            ''' for a given nr_ finds the r that obtains nr_'''
            z = (r-mu)/sigma
            zetta = sts.norm.pdf(z) - z*(1-sts.norm.cdf(z))
            nr = zetta*sigma
            return np.abs(nr-nr_)

        def get_nr(r):
            ''' for a given x="r" obtains n(r). basically x is self.r_new[cls]'''
            z = (r-mu)/sigma
            zetta = sts.norm.pdf(z) - z*(1-sts.norm.cdf(z))
            nr = zetta*sigma
            return nr

        def get_n2r(r):
            ''' for a given x="r" obtains n(r). basically x is self.r_new[cls]'''
            z = (r-mu)/sigma
            zetta = 0.5*((z*z+1)*(1-sts.norm.cdf(z)) - z*sts.norm.pdf(z))
            n2r = zetta*sigma*sigma
            return n2r

        epsilon = 0.001
        max_r = 2000
        n2r = 1
        if self.clusters == 1:
            self.QP_new = np.sqrt(2*(self.K*self.lambdaa + 
                                        (self.ch + self.cp)*n2r/self.ch))*np.ones(self.clusters)
            self.rP_new = np.zeros(self.clusters)
        else:
            self.QP_new = np.squeeze(np.sqrt(2*(self.K*self.lambdaa + 
                                        (self.ch + self.cp)*n2r/self.ch))*np.ones((1,self.clusters)))
            self.rP_new = np.squeeze(np.zeros((1,self.clusters)))

        for cls in range(self.clusters):
            notStop = True
            i = 0    
            #print "(r,Q)= (%0.2f" %(self.rP_new[cls]),", %0.2f )" %(self.QP_new[cls])   
            max_r = sts.norm.isf(1-(self.cp/(self.cp+self.ch+.0)), self.mu[cls], self.sigma[cls])
            while notStop:
                i += 1
        #         print "iteration ", i
                # reset the value of r,Q
                self.QP_old = self.QP_new[cls]
                self.rP_old = self.rP_new[cls]

                # update r
                self.nr_ = (self.QP_old*self.ch)/(self.cp + self.ch)
                if self.sigma[cls] < 1e-8:
                    self.sigma[cls] = 1e-8  

                sigma = self.sigma[cls]
                mu = self.mu[cls]
                nr_ = self.nr_
                res = minimize_scalar(solve_nr, method='Golden', bracket=(0,max_r))
                # get_nr(res.x) -> gives the n(r) of the obtained r
                self.rP_new[cls] = res.x

                # update Q
                self.n2r = get_n2r(self.rP_new[cls])
                self.QP_new[cls] = np.sqrt(2*(self.K*self.lambdaa + 
                                                  (self.ch + self.cp)*self.n2r)/self.ch)
                #print "nr= %0.2f" %nr_,", n2r= %0.2f" %self.n2r,", (r,Q)= (%0.2f" %(self.rP_new[cls]),", %0.2f )" %(self.QP_new[cls])   
                #print "(r,Q)= (%0.2f" %(self.rP_new[cls]),", %0.2f )" %(self.QP_new[cls])   
                # check if we should stop 
                if np.abs(self.QP_new[cls]-self.QP_old) < epsilon:
                    if np.abs(self.rP_new[cls]-self.rP_old) < epsilon:
                        notStop = False
            if self.if_print_aprx_rq_solutions:
                print cls, '( %0.2f' %(self.rP_new[cls]) ,', %0.2f' %self.QP_new[cls], ')', 'g(r,Q) is: %0.2f' \
                %(self.ch*(self.rP_new[cls] - self.lambdaa*self.l[cls] + self.QP_new[cls]/2) +
                self.K*self.lambdaa/self.QP_new[cls] +
                (self.cp + self.ch)*self.lambdaa*self.n2r/self.QP_new[cls]) , i, 'iterations'


    def variable_builder_rand(self, shape):
        return tf.Variable(np.random.normal(0, self.var, shape))

    def variable_builder_fix(self, liste):
        result_list = [] 
        for i in liste:
            result_list += [tf.Variable(i)]
        return result_list

    # creat dnn training method 
    def create_train_method(self):
        # import the data
        #from tensorflow.examples.tutorials.mnist import input_data
        # placeholders, which are the training data

        self.x = tf.placeholder(tf.float64, shape=[None,self.input_dim], name='x')
        self.y_ = tf.placeholder(tf.float64, shape=[None], name='y_')
        self.init_IL = tf.placeholder(tf.float64, shape=[None,1], name='init_IL')
        self.learning_rate = tf.placeholder(tf.float64, shape=[], name='lr')
        self.zero = tf.placeholder(tf.float64, shape=[None,1], name='zero')
        self.c_h = tf.placeholder(tf.float64, shape=[None,1], name='c_h')
        self.c_p = tf.placeholder(tf.float64, shape=[None,1], name='c_p')
        self.K_ = tf.placeholder(tf.float64, shape=[None,1], name='K')
        self.L_ = tf.placeholder(tf.float64, shape=[None,1], name='L')
        self.lambdaa_ = tf.placeholder(tf.float64, shape=[None,1], name='lambdaa')

        self.w=[]
        self.b=[]
        self.best_w = []
        self.best_b = []
        self.layer = []
        self.y=0
        if self.run_number != 0:
            self.w = self.variable_builder_fix(ww)
            self.b = self.variable_builder_fix(bb)
        # define the variables
        for j in range(self.NoHiLay+1):

            if self.run_number == 0:
                self.w += [self.variable_builder_rand([self.nodes[j], self.nodes[j+1]])]
                self.b += [self.variable_builder_rand(self.nodes[j+1])]      
            if j == 0:
                if self.ifRelu:
                    self.layer += [tf.nn.relu(tf.matmul(self.x, self.w[j]) + self.b[j])]
                else:
                    self.layer += [tf.nn.sigmoid(tf.matmul(self.x, self.w[j]) + self.b[j])]
            elif j == self.NoHiLay:
                self.y = tf.matmul(self.layer[j-1], self.w[j]) + self.b[j]
            else:
                self.layer += [tf.nn.sigmoid(tf.matmul(self.layer[j-1], self.w[j]) + self.b[j])]

        # Passing global_step to minimize() will increment it at each step.
        self.global_step = tf.Variable(0, trainable=False)
        self.momentum = tf.Variable(self.init_momentum, trainable=False)

        # r=y[:,0]; Q=y[:,1]
        # prediction function (just one layer)
        if self.rQ:    
            self.diff = tf.subtract(self.y_, self.y[:,0])
            # tf.greater(x,y) returns the truth value of (x > y) element-wise.e.g. [[False False  True]]
            self.result = tf.greater(self.zero[:,0],self.diff)
            # force the value of r be positive 
            self.r_positive = tf.greater(self.y[:,0], self.zero[:,0])
            self.r_negative = tf.greater(self.zero[:,0], self.y[:,0])
            self.r = tf.where(self.r_positive, self.y[:,0], self.zero[:,0])
            self.r_penalty = tf.where(self.r_negative, self.y[:,0], self.zero[:,0])
            # condition on the cost function 
            if self.EIL:
                # get n(r)=(d-r)^+
                self.nr = tf.where(self.result, self.zero[:,0], self.diff)
                # tf.select(condition, t, e) : output should be taken from t (if true) or e (if false).
                self.c = tf.where(self.result, self.zero[:,0], self.c_p[:,0])
                # cost function
                # tf.mul returns x * y element-wise.
                self.cost_function = tf.reduce_sum(
                             tf.multiply(self.c_h[:,0], self.r - tf.multiply(self.lambdaa_[:,0],self.L_[:,0]) 
                                       + tf.scalar_mul(0.5, self.y[:,1]))
                                       + tf.divide(tf.multiply(self.lambdaa_[:,0], self.K_[:,0]), self.y[:,1])
                                       + tf.multiply(tf.divide(tf.multiply(self.lambdaa_[:,0],self.c), self.y[:,1]),
                                                tf.abs(self.nr)))
                self.c1 = tf.multiply(self.c_h[:,0], self.r - 
                                tf.multiply(self.lambdaa_[:,0],self.L_[:,0])+ tf.scalar_mul(0.5, self.y[:,1]))
                self.c2 = tf.divide(tf.multiply(self.lambdaa_[:,0], self.K_[:,0]), self.y[:,1])
                self.c3 = tf.multiply(tf.divide(tf.multiply(self.lambdaa_[:,0], self.c), self.y[:,1]),
                                      tf.abs(self.nr))
                self.single_cost_function = self.c1 + self.c2 + self.c3

                self.surrogate_cost_function = self.cost_function + 100*tf.reduce_sum(tf.abs(self.r_penalty))
            elif self.aprx:
                # get (d-r)^+
                self.dmr = tf.where(self.result, self.zero[:,0], self.diff)
                # get n^2(r)= 0.5*((d-r)^+)^2
                self.nr = tf.scalar_mul(0.5, tf.multiply(self.dmr, self.dmr))
                # tf.select(condition, t, e) : output should be taken from t (if true) or e (if false).
                self.c = tf.where(self.result, self.zero[:,0], self.c_p[:,0] + self.c_h[:,0])
                # cost function
                # tf.mul returns x * y element-wise.
                self.cost_function = tf.reduce_sum(
                             tf.multiply(self.c_h[:,0], self.r - tf.multiply(self.lambdaa_[:,0],self.L_[:,0]) 
                                       + tf.scalar_mul(0.5, self.y[:,1]))
                                       + tf.divide(tf.multiply(self.lambdaa_[:,0], self.K_[:,0]), self.y[:,1])
                                       + tf.multiply(tf.divide(self.c, self.y[:,1]),
                                                self.nr))
                self.c1 = tf.multiply(self.c_h[:,0], self.r - 
                                tf.multiply(self.lambdaa_[:,0],self.L_[:,0])+ tf.scalar_mul(0.5, self.y[:,1]))
                self.c2 = tf.divide(tf.multiply(self.lambdaa_[:,0], self.K_[:,0]), self.y[:,1])
                self.c3 = tf.multiply(tf.divide(self.c, self.y[:,1]), self.nr)
                self.single_cost_function = self.c1 + self.c2 + self.c3

                self.surrogate_cost_function = self.cost_function + 100*tf.reduce_sum(tf.abs(self.r_penalty))

        # the newsvendor problem 
        else:
            self.diff = tf.subtract(self.y_, self.y[:,0])
            # tf.greater(x,y) returns the truth value of (x > y) element-wise.e.g. [[False False  True]]
            self.result = tf.greater(self.zero[:,0], self.diff)
            # tf.select(condition, t, e) : output should be taken from t (if true) or e (if false).    
            self.c = tf.where(self.result, self.c_h[:,0], self.c_p[:,0])
            # cost function
            # tf.mul returns x * y element-wise.
            if (loss_type is 'L2'):
                self.cost_function = tf.reduce_mean(tf.square(tf.multiply(self.diff, self.c)))/2
            elif (loss_type is 'L1'):
                self.cost_function = tf.reduce_mean(tf.abs(tf.multiply(self.diff, self.c)))/2

            self.nw_cost_function = tf.reduce_sum(tf.abs(tf.multiply(self.diff, self.c)))

        self.l2regularization = 0
        for j in range(self.NoHiLay+1):  
            self.l2regularization += tf.reduce_sum(tf.square(self.w[j])) + tf.reduce_sum(tf.square(self.b[j])) 
        if self.rQ:
            self.loss = self.surrogate_cost_function + self.config.l2lambda*self.l2regularization
        else:
            self.loss = self.cost_function + self.config.l2lambda*self.l2regularization

        # define the training paramters and model, gradient model and feeding the function
        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss, global_step=global_step)
        learning_rate = tf.train.exponential_decay(self.config.rl0, self.global_step,
                                           self.config.decay_step, self.config.decay_rate_stair, staircase=True)                
        self.train_step = tf.train.AdamOptimizer(learning_rate,0.9,0.999,1e-8).minimize(
                                self.loss, global_step=self.global_step)
        # initilize the variables
        self.saver = tf.train.Saverconfig.()
        
        self.sess.run(tf.global_variables_initializer())
        self.iter = 0
        

    def run_dnn(self):
    ''' run the dnn method for a given number of iterations, print the results, 
         and holds them in self.train_result, self.val_result, self.test_result'''
        self.train_result = []
        self.test_result = []
        self.val_result = []
        # Train the Model for 1000 times. by defining the batch number we determine that it is sgd
        if self.rQ:
            for i in range(self.config.maxiter):
                self.iter += 1 
                batch = np.random.randint(0,self.train_size, size=self.config.batch_size) 
#                 lr = self.config.rl0
                lr = self.config.rl0*np.power(1 + self.config.decay_rate*(i+1), -self.power)
                self.sess.run(self.train_step, feed_dict={self.x:self.train_x[batch], 
                            self.learning_rate:lr, self.c_p:self.shrtg_cost_tr[batch],
                            self.c_h:self.hld_cost_tr[batch], self.zero:self.zeros_tr[batch], 
                              self.y_:self.train_y[batch], self.K_:self.order_cost_tr[batch]
                             , self.L_:self.l_tr[batch], self.lambdaa_:self.lambdaa_tr[batch]})
                if np.mod(i, self.config.display) == 0:
                    feed_dict={self.x: self.train_x, self.y_: self.train_y, self.c_p:self.shrtg_cost_tr, 
                               self.c_h:self.hld_cost_tr, self.zero:self.zeros_tr
                              , self.K_:self.order_cost_tr, self.L_:self.l_tr , self.lambdaa_:self.lambdaa_tr}

                    self.train_result += [self.sess.run(self.cost_function, feed_dict)]
                # validation
                    feed_dict={self.x: self.valid_x, self.y_: self.valid_y, self.c_p:self.shrtg_cost_val, 
                               self.c_h:self.hld_cost_val, self.zero:self.zeros_val, 
                              self.K_:self.order_cost_val, self.L_:self.l_val, self.lambdaa_:self.lambdaa_val}
                    self.val_result += [self.sess.run(self.cost_function,feed_dict)]

                    print ("Iter" , i, "lr %.6f" %lr , "| Train %.2f" %self.train_result[-1] , 
                           "| Validation %.2f" %self.val_result[-1] , "||W|| %.2f" %(self.sess.run(self.l2regularization)), 
                           "lmbd*||W|| %.2f" %(self.config.l2lambda*self.sess.run(self.l2regularization))
                          )
                    if self.if_print_rQ:
                        out = self.sess.run([self.cost_function, self.y],feed_dict)
                        r = np.unique(out[1][:,0])
                        q = np.unique(out[1][:,1])
                        print "(r,Q)= ", r, q
                    
                    if self.if_print_rQ_test_set:
                        print ""
                        print self.sess.run(self.y,feed_dict={self.x: self.train_x})
                        print self.sess.run(self.y,feed_dict={self.x: self.valid_x})

            self.saver.save(self.sess, self.model_dir+'/model', global_step=self.iter)
            print "network weights are saved"


        else:
            for i in range(self.config.maxiter):
                batch = np.random.randint(0,self.train_size, size=self.config.batch_size) 
                lr = self.config.rl0*np.power(1 + self.config.decay_rate*(i+1), -self.power)
                self.sess.run(train_step, feed_dict={self.x:self.train_x[batch], self.y_:self.train_y[batch] , 
                                        self.learning_rate:lr, self.c_p:self.shrtg_cost_tr[batch],
                                        self.c_h:self.hld_cost_tr[batch], self.zero:self.zeros_tr[batch]})
                if np.mod(i, self.config.display) == 0:
                    feed_dict={self.x: self.train_x, self.y_: self.train_y, self.c_p:self.shrtg_cost_tr, 
                               self.c_h:self.hld_cost_tr, self.zero:self.zeros_tr}
                    self.train_result = self.sess.run(self.cost_function, feed_dict)
                    feed_dict={self.x: self.valid_x, self.y_: self.valid_y, self.c_p:self.shrtg_cost_val, 
                               self.c_h:self.hld_cost_val, self.zero:self.zeros_val}
                    self.valid_result = self.sess.run(self.cost_function, feed_dict)
                    print ("Iter" , i, "lr %.6f" %lr , "| Train %.2f" %self.train_result ,
                           "| Test %.2f" %self.valid_result , 
                    "||W|| %.2f" %(self.sess.run(self.l2regularization)),
                           "lmbd*||W|| %.2f" %(l2lambda*self.sess.run(self.l2regularization)) 
                          )
                    
    def print_EIL_DNN_costs(self):
        ''' get optimal solutions, prints the EIL and DNN cost and the improvement of DNN over EIL '''
        print 'optimal solutions'
        cost_val = self.get_eil_cost(self.valid_y, self.ind_valid)
        cost_tr = self.get_eil_cost(self.train_y, self.ind_train)
        print '\t   optimal' , '\t', '   DNN'
        print 'train \t ', "%.1f" %cost_tr, '\t', "%.1f" %self.train_result[-1],\
                '\t', "%.3f" %((self.train_result[-1] - cost_tr)/cost_tr)
        print 'valid \t ', "%.1f" %cost_val, '\t', "%.1f" %self.val_result[-1],\
                '\t', "%.3f" %((self.val_result[-1] - cost_val)/cost_val)
        print 'avg train real cost',  cost_tr/self.train_size, ',avg validation real cost', \
            cost_val/self.valid_size           

    def print_EIL_DNN_KNN_costs(self):
        ''' get optimal solutions '''
        print 'optimal solutions'
        cost_val = self.get_eil_cost(self.valid_y, self.ind_valid)
        cost_te = self.get_eil_cost(self.test_y, self.ind_test)
        cost_tr = self.get_eil_cost(self.train_y, self.ind_train)
        print '\t  \t  \t  optimal' , '\t', '     DNN' , '\t', '         KNN'
        print self.dis,  self.clusters, 'train \t ', "%.1f" %cost_tr, '\t', "%.1f" %self.train_result[-1],\
                '\t 0 \t', "%.3f" %((self.train_result[-1] - cost_tr)/cost_tr)
        print self.dis,  self.clusters, 'valid \t ', "%.1f" %cost_val, '\t', "%.1f" %self.val_result[-1],\
                '\t 0 \t', "%.3f" %((self.val_result[-1] - cost_val)/cost_val)
        print self.dis,  self.clusters, 'test \t ', \
        "%.1f" %cost_te, '\t', "%.1f" %self.test_result[-1],\
                '\t', "%.1f" %(self.g_knn_total), \
            '\t', "%.3f" %((self.test_result[-1] - cost_te)/cost_te)
        print 'avg train real cost',  cost_tr/self.train_size, ',avg validation real cost', \
            cost_val/self.valid_size, ',avg test real cost', cost_te/self.test_size                    

    
    def set_knn_settings(self, k_):
    '''set k in knn algorithm '''
        self.k_ = k_

    def knn_saa(self):
    '''runs a knn based on saa to obtain (r,Q) and it reports r_knn and Q_knn in the end. '''
        big_r = 1000
        zero_ = np.zeros(self.k_)
        nbrs = NearestNeighbors(n_neighbors=self.k_, algorithm='ball_tree').fit(self.train_x)
        _, knn_te = nbrs.kneighbors(self.test_x)
        _, knn_val = nbrs.kneighbors(self.valid_x)

        epsilon = 0.1
        self.g_knn_total = 0
        self.Q_knn = np.squeeze(np.sqrt(2*self.K*self.lambdaa/self.ch)*np.ones((1,self.test_size)))
        self.r_knn = np.squeeze(10000*np.ones((1,self.test_size)))
        if len(np.shape(self.Q_knn)) == 0:
            self.Q_knn = np.expand_dims(self.Q_knn,0)
        if len(np.shape(self.r_knn)) == 0:
            self.r_knn = np.expand_dims(self.r_knn,0)
        # print '(r,Q) is: ', r_knn, Q_knn
        for te in range(self.test_size):
            notStop = True
            i = 0    
            while notStop:
                i += 1
                # print "iteration ", i
                # reset the value of r,Q
                Q_old = self.Q_knn[te]
                r_old = self.r_knn[te]
                # update r
                z = (Q_old*self.ch)/(self.cp*self.lambdaa)
                if self.clusters == 1:
                    index = 0 
                else:
                    index = self.ind_test[te,0]
                if z >= 0 and z <= 1:
                    if self.sigma[int(index)] <= 1e-16:
                        self.r_knn[te] = self.mu[int(index)]
                    else:
                        self.r_knn[te] = sts.norm.isf(z,self.mu[int(index)],self.sigma[int(index)]) # isf works with 1-cdf 
                else:
                    self.r_knn[te] = -big_r
                # update Q
                # this nr is based on KNN-SAA        
                nr = 1/(.0 + self.k_)*sum(np.maximum(self.train_y[knn_te[te]] - self.r_knn[te], zero_))

                #nr = zetta*self.sigma[int(index)]
                self.Q_knn[te] = np.sqrt((2*self.lambdaa*(self.K+self.cp*nr))/self.ch)
                # check if we should stop 
                if np.abs(self.Q_knn[te]-Q_old) < epsilon:
                    if np.abs(self.r_knn[te]-r_old) < epsilon:
                        notStop = False
            g = self.ch*(self.r_knn[te] - self.lambdaa*self.l[int(index)] + self.Q_knn[te]/2) + \
                self.K*self.lambdaa/self.Q_knn[te] + self.cp*self.lambdaa*nr/self.Q_knn[te]
            self.g_knn_total += g 
            if self.config.if_print_knn_solutions:
                print te, '(', self.r_knn[te] ,',', self.Q_knn[te], ')', 'g(r,Q) is: ', g , i, 'iterations'
        if self.config.if_print_knn_final_cost:
            print self.g_knn_total

        
    def restore_dnn(self):
        ''' restore a saved dnn network'''
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path

        else:
            print "Could not find old network weights in", self.model_dir
            
    def call_dnn_fp(self):
        ''' calls the forward pass of train, validation, and the test datasets'''
        self.rq_test = self.sess.run(self.y,feed_dict={self.x: self.test_x})
        self.test_result = []
        self.train_result = []
        self.val_result = []

        feed_dict={self.x: self.train_x, self.y_: self.train_y, self.c_p:self.shrtg_cost_tr, 
                   self.c_h:self.hld_cost_tr, self.zero:self.zeros_tr
                  , self.K_:self.order_cost_tr, self.L_:self.l_tr , self.lambdaa_:self.lambdaa_tr}

        self.train_result += [self.sess.run(self.cost_function, feed_dict)]
        # validation
        feed_dict={self.x: self.valid_x, self.y_: self.valid_y, self.c_p:self.shrtg_cost_val, 
                   self.c_h:self.hld_cost_val, self.zero:self.zeros_val, 
                  self.K_:self.order_cost_val, self.L_:self.l_val, self.lambdaa_:self.lambdaa_val}
        self.val_result += [self.sess.run(self.cost_function,feed_dict)]


        feed_dict={self.x: self.test_x, self.y_: self.test_y, self.c_p:self.shrtg_cost_te, 
                   self.c_h:self.hld_cost_te, self.zero:self.zeros_te, 
                  self.K_:self.order_cost_te, self.L_:self.l_te, self.lambdaa_:self.lambdaa_te}
        self.test_result += [self.sess.run([self.cost_function, self.y],feed_dict)]
       
    
    def get_rq_test(self, x_input, binary_x=False):
        ''' x =[day,month,department] is the non-binary input value 
        x_input = [[4,5,12], [1,5,2]] 
        x_input=[[4,5,12]] '''

        if not binary_x:
            binary_day = []
            binary_month = []
            binary_department_24 = []
            binary_department_29 = []

            A = np.identity(7)
            for i in range(7):
                    binary_day += [A[i : i + 1]]

            A = np.identity(12)
            for i in range(12):
                    binary_month += [A[i : i + 1]]

            A = np.identity(24)
            for i in range(24):
                binary_department_24 += [A[i : i + 1]] 

            A = np.identity(29)
            for i in range(29):
                binary_department_29 += [A[i : i + 1]]         

            x_binary = []

            for x in x_input:
                if self.clusters == 1:
                    x_binary += [np.ones(1)]
                elif self.clusters == 10:
                    x_binary += [np.concatenate((binary_day[x[0]%2 + 1] , binary_department_24[x[2]%5 + 1]) , axis = 1)]
                elif self.clusters == 100:
                    x_binary += [np.concatenate((binary_day[x[0]%5 + 1] , binary_department_24[x[2]%19 + 1]) , axis = 1)]
                elif self.clusters == 200 or self.real_cluster == 103:
                    x_binary += [np.concatenate((binary_day[x[0]%7] , binary_department_29[x[2]%29]) , axis = 1)]

            x_binary = np.squeeze(np.array(x_binary),axis=1)
        else:
            x_binary = x_input
        # print np.shape(x_binary)
        feed_dict={self.x: x_binary}
        test_y = self.sess.run(self.y, feed_dict)

        return (test_y)      
    

    def set_network(self):
        ''' set the best find network structure for EIL cost function'''
        if self.clusters == 1 and self.dis == 'normal':
            self.nodes = [1, 1, 0, 2]
        elif self.clusters == 10 and self.dis == 'normal':
            self.nodes = [31, 22, 20, 2]
        elif self.clusters == 100 and self.dis == 'normal':
            self.nodes = [31, 45, 76, 67, 2] 
        elif self.clusters == 200 and self.dis == 'normal':
            self.nodes = [36, 50, 84, 46, 2]

        elif self.clusters == 1 and self.dis == 'beta':
            self.nodes = [1, 2, 2, 1, 2]    
        elif self.clusters == 10 and self.dis == 'beta':
            self.nodes = [31, 63, 43, 2]
        elif self.clusters == 100 and self.dis == 'beta':
            self.nodes = [31, 45, 76, 67, 2]
        elif self.clusters == 200 and self.dis == 'beta':
            self.nodes = [36, 93, 87, 45, 2]

        elif self.clusters == 1 and self.dis == 'lognormal':
            self.nodes = [1, 1, 0, 2]
        elif self.clusters == 10 and self.dis == 'lognormal':
            self.nodes = [31, 61, 59, 2]
        elif self.clusters == 100 and self.dis == 'lognormal':
            self.nodes = [31, 43, 74, 39, 2]
        elif self.clusters == 200 and self.dis == 'lognormal':
            self.nodes = [36, 64, 42, 2]

        elif self.clusters == 1 and self.dis == 'exponential':
            self.nodes = [1, 2, 1, 2]    
        elif self.clusters == 10 and self.dis == 'exponential':
            self.nodes = [31, 15, 10, 2]
        elif self.clusters == 100 and self.dis == 'exponential':
            self.nodes = [31, 56, 29, 14, 2]
        elif self.clusters == 200 and self.dis == 'exponential':
            self.nodes = [36, 70, 38, 2]

        elif self.clusters == 1 and self.dis == 'uniform':
            self.nodes = [1, 2, 2, 1, 2]    
        elif self.clusters == 10 and self.dis == 'uniform':
            self.nodes = [31, 77, 49, 2]
        elif self.clusters == 100 and self.dis == 'uniform':
            self.nodes = [31, 24, 22, 2]
        elif self.clusters == 200 and self.dis == 'uniform':
            self.nodes = [36, 52, 71, 49, 2]

        self.NoHiLay = len(self.nodes) - 2        

    def create_mem(self):
        ''' create the required memory to run the simulation for dnn, knn, and eil algorithms'''
        self.dnn_cost = np.zeros((self.items, self.config.sim_period+1))
        self.knn_cost = np.zeros((self.items, self.config.sim_period+1))
        self.eil_cost = np.zeros((self.items, self.config.sim_period+1))

        self.dnn_rQ = np.zeros((self.items, self.config.sim_period+1,2))
        self.knn_rQ = np.zeros((self.items, self.config.sim_period+1,2))
        self.eil_rQ = np.zeros((self.items, self.config.sim_period+1,2))

        self.IL_dnn = np.zeros((self.items, self.config.sim_period+1))
        self.IL_knn = np.zeros((self.items, self.config.sim_period+1))
        self.IL_eil = np.zeros((self.items, self.config.sim_period+1))

        self.AO_init= 0
        self.AO_dnn = self.AO_init*np.ones((self.items, self.config.sim_period+1+int(365*max(self.l))))
        self.AO_knn = self.AO_init*np.ones((self.items, self.config.sim_period+1+int(365*max(self.l))))
        self.AO_eil = self.AO_init*np.ones((self.items, self.config.sim_period+1+int(365*max(self.l))))

    def run_simulator(self):
        ''' This function loads the data and run a (r,Q) simulator to 
        compare knn, dnn, and eil. For DNN, it loads the pre-trained saved models, 
        but for knn and eil it calls them on the fly, so knn is quite expensive to run. 
        To load the dnn, it uses the function set_network, which has the address and info
        of the best found network and hyper-parameters.
        '''
        try:
            self.r_new
        except:    
            self.if_print_eil_solutions = False
            self.get_EIL_solution()

        if self.clusters == 1:
            self.items = 1
        elif self.clusters == 10:
            self.items = 5
        elif self.clusters == 100:
            self.items = 24
        elif self.clusters == 200:
            self.items = 29


        # get the list of unique inds
        self.indexes = sorted([list(x) for x in set(tuple(x) for x in self.ind_test)], key=lambda k: k[0])

        # build the pandas dataframe of the test (ind,y,x)
        test_data_pd = pd.DataFrame(np.concatenate((self.ind_test, np.expand_dims(self.test_y,1), self.test_x), axis=1))

        # get a dictionary with the index of ind 
        self.test_data_dic = {}
        for i in range(self.clusters):
            if self.clusters == 1:
                self.test_data_dic[i] = test_data_pd[test_data_pd[0]==1].values
            else:
                self.test_data_dic[i] = test_data_pd[test_data_pd[0]==i].values


        if not self.config.use_current_trained_network:
            tf.reset_default_graph()

            # create the dnn model  
            self.set_dnn_settings()
            # revide the network size based on the current instance
            self.set_network()
            cur_dir=os.path.join('rq/best_networks/', self.dis)
            self.model_dir=os.path.join(cur_dir, str(self.clusters))

            # create the train method and restore the network
            self.create_train_method()
            self.restore_dnn()

        self.create_mem()

        dnn_time = 0
        knn_time = 0

        if self.clusters == 1:
            self.no_days = 1
        elif self.clusters == 10:
            self.no_days = 2                    
        elif self.clusters == 100 or self.clusters == 200:
            self.no_days = 7                    

        # use this saved_index later to use in EIL algorithm. 
        self.saved_ind = np.zeros((self.no_days,self.items))
        for i in range(self.clusters):
            self.saved_ind[i%self.no_days, i%self.items] = i


        obs_count = 0
        for i in range(self.items):
            for t in range(self.config.sim_period):

                fixed_cost_dnn = 0
                fixed_cost_knn = 0
                fixed_cost_eil = 0
                # set the time_ index 
                if self.clusters == 1:
                    ind = 0          
                elif self.clusters == 10:
                    time_ = t%2
                    ind = int(self.saved_ind[t%self.no_days, i%self.items])                    
                    # ind = (i*2 + time_)%self.clusters
                elif self.clusters == 100 or self.clusters == 200:
                    time_ = t%7
                    self.no_days = 7                    
                    ind = int(self.saved_ind[t%self.no_days, i%self.items])
                    # ind = (i*7 + time_)%self.clusters

                # take the 0th row of the current dataset, obtain x,y; and then delete the 0th row 
                if len(self.test_data_dic[ind]) == 0:
                    continue 
                if self.dis == 'exponential':
                    y = self.test_data_dic[ind][0][2]
                    x_input = np.expand_dims(self.test_data_dic[ind][0][3:],0)
                    # set the test data set with new sample 
                    self.ind_test = np.expand_dims(self.test_data_dic[ind][0][0:2],0)
                    self.test_x = x_input
                    demand = self.test_data_dic[ind][0][2]
                else:
                    y = self.test_data_dic[ind][0][3]
                    x_input = np.expand_dims(self.test_data_dic[ind][0][4:],0)
                    # set the test data set with new sample 
                    self.ind_test = np.expand_dims(self.test_data_dic[ind][0][0:3],0)
                    self.test_x = x_input
                    demand = self.test_data_dic[ind][0][3]
                    
                self.test_data_dic[ind] = np.delete(self.test_data_dic[ind],0,axis=0)

                # call dnn forward pass to get r,Q
                start_t = time.time()
                self.dnn_rQ[i,t,:] = np.squeeze(self.get_rq_test(x_input, binary_x=True))
                dnn_time += time.time() - start_t

                # call eil and knn to get the cost
                self.eil_rQ[i,t,:] = [self.r_new[ind], self.Q_new[ind]]
                self.if_print_knn_solutions = False
                self.set_knn_settings(10)
                self.test_size = 1

                start_t = time.time()
                self.knn_saa()
                self.knn_rQ[i,t,:] = [self.r_knn, self.Q_knn]
                knn_time += time.time() - start_t

                self.IL_dnn[i,t] = self.IL_dnn[i,t] - demand
                self.IL_knn[i,t] = self.IL_knn[i,t] - demand
                self.IL_eil[i,t] = self.IL_eil[i,t] - demand

                # L is the lead time_, in scale of a year
                L = int(365*self.l[ind])
                if sum(self.AO_dnn[i,t:]) + self.IL_dnn[i,t] < self.dnn_rQ[i,t,0]:
                    self.AO_dnn[i,t+L] += self.dnn_rQ[i,t,1]
                    fixed_cost_dnn = self.K
                if sum(self.AO_knn[i,t:]) + self.IL_knn[i,t] < self.knn_rQ[i,t,0]:
                    self.AO_knn[i,t+L] += self.knn_rQ[i,t,1]
                    fixed_cost_knn = self.K
                if sum(self.AO_eil[i,t:]) + self.IL_eil[i,t] < self.eil_rQ[i,t,0]:
                    self.AO_eil[i,t+L] += self.eil_rQ[i,t,1]
                    fixed_cost_eil = self.K

                self.dnn_cost[i,t] = max(self.IL_dnn[i,t],0)*self.ch + max(-self.IL_dnn[i,t],0)*self.cp \
                                        + fixed_cost_dnn
                self.knn_cost[i,t] = max(self.IL_knn[i,t],0)*self.ch + max(-self.IL_knn[i,t],0)*self.cp \
                                        + fixed_cost_knn
                self.eil_cost[i,t] = max(self.IL_eil[i,t],0)*self.ch + max(-self.IL_eil[i,t],0)*self.cp \
                                        + fixed_cost_eil

                self.IL_dnn[i,t+1] = self.IL_dnn[i,t] + self.AO_dnn[i,t]
                self.IL_knn[i,t+1] = self.IL_knn[i,t] + self.AO_knn[i,t]
                self.IL_eil[i,t+1] = self.IL_eil[i,t] + self.AO_eil[i,t]     

                if t%20 == 0: 
                    print t, "periods passed of item ", i
        print "dnn \t", "knn \t", "eil \t"
        print self.dis, self.clusters, "%0.2f" %(sum(np.sum(self.dnn_cost, axis=1))), \
            "%0.2f" %(sum(np.sum(self.knn_cost, axis=1))), \
            "%0.2f" %(sum(np.sum(self.eil_cost, axis=1)))
        print self.dis, "mean", self.clusters, "%0.2f" %(np.mean(self.dnn_cost)), \
            "%0.2f" %(np.mean(self.knn_cost)), \
            "%0.2f" %(np.mean(self.eil_cost))
        print self.dis, "std", self.clusters, "%0.2f" %(np.std(self.dnn_cost)), \
            "%0.2f" %(np.std(self.knn_cost)), \
            "%0.2f" %(np.std(self.eil_cost))
        print self.dis, "obs_count", self.clusters, "%d" %(obs_count) 
        print self.dis, self.clusters, "cpu times are %0.2f" %(dnn_time), \
            "%0.2f" %(knn_time)        


    def get_network(self, net_num):
        ''' This function gets the network structure of a given saved model and 
        returns it in "w". ''' 
        # import checkpoint_utils as checkpoint_utils
        from tensorflow.contrib.framework.python.framework import checkpoint_utils
        self.model_dir = '/scratch/afo214/newsvendor/rq_runner_code/saved_networks/'
        self.model_dir = os.path.join(self.model_dir, self.dis)
        self.model_dir = os.path.join(self.model_dir, str(self.clusters))
        self.model_dir = os.path.join(self.model_dir, str(net_num))
        # print self.model_dir
        update_checkpoint(net_num, self.model_dir)
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        w =[]
        cnt = 0 
        if checkpoint and checkpoint.model_checkpoint_path:
            var_list = checkpoint_utils.list_variables(self.model_dir)
            for v in var_list:
                if 'Adam_' in v[0]:
                    print v            
                    if cnt == 0:
                        w += [v[1][0]]
                    if len(v[1]) == 1:
                        w += v[1]
                    cnt += 1
        return w 

    def run_simulator_select_best(self, net_num):
        ''' This function loads the data and run a (r,Q) simulator to 
        get the cost of a given pre-trained saved dnn models which in runner had number net_num. 
        '''
        dnn_time = 0

        if self.clusters == 1:
            self.items = 1
        elif self.clusters == 10:
            self.items = 5
        elif self.clusters == 100:
            self.items = 24
        elif self.clusters == 200:
            self.items = 29


        # get the list of unique inds
        self.indexes = sorted([list(x) for x in set(tuple(x) for x in self.ind_test)], key=lambda k: k[0])

        # build the pandas dataframe of the test (ind,y,x)
        test_data_pd = pd.DataFrame(np.concatenate((self.ind_test, np.expand_dims(self.test_y,1), self.test_x), axis=1))

        # get a dictionary with the index of ind 
        self.test_data_dic = {}
        for i in range(self.clusters):
            if self.clusters == 1:
                self.test_data_dic[i] = test_data_pd[test_data_pd[0]==1].values
            else:
                self.test_data_dic[i] = test_data_pd[test_data_pd[0]==i].values


        tf.reset_default_graph()

        # create the dnn model  
        self.set_dnn_settings()
        # revide the network size based on the current instance
        self.nodes=self.get_network(net_num)
        self.NoHiLay = len(self.nodes) - 2        
        # create the train method and restore the network
        self.create_train_method()
        self.restore_dnn()

        self.create_mem()

        if self.clusters == 1:
            self.no_days = 1
        elif self.clusters == 10:
            self.no_days = 2                    
        elif self.clusters == 100 or self.clusters == 200:
            self.no_days = 7                    

        # use this saved_index later to use in EIL algorithm. 
        self.saved_ind = np.zeros((self.no_days,self.items))
        for i in range(self.clusters):
            self.saved_ind[i%self.no_days, i%self.items] = i


        obs_count = 0
        for i in range(self.items):
            for t in range(self.config.sim_period):

                fixed_cost_dnn = 0
                # set the time_ index 
                if self.clusters == 1:
                    ind = 0          
                elif self.clusters == 10:
                    time_ = t%2
                    ind = int(self.saved_ind[t%self.no_days, i%self.items])                    
                    # ind = (i*2 + time_)%self.clusters
                elif self.clusters == 100 or self.clusters == 200:
                    time_ = t%7
                    self.no_days = 7                    
                    ind = int(self.saved_ind[t%self.no_days, i%self.items])
                    # ind = (i*7 + time_)%self.clusters

                # take the 0th row of the current dataset, obtain x,y; and then delete the 0th row 
                if len(self.test_data_dic[ind]) == 0:
                    continue 
                if self.dis == 'exponential':
                    y = self.test_data_dic[ind][0][2]
                    x_input = np.expand_dims(self.test_data_dic[ind][0][3:],0)
                    # set the test data set with new sample 
                    self.ind_test = np.expand_dims(self.test_data_dic[ind][0][0:2],0)
                    self.test_x = x_input
                    demand = self.test_data_dic[ind][0][2]
                else:
                    y = self.test_data_dic[ind][0][3]
                    x_input = np.expand_dims(self.test_data_dic[ind][0][4:],0)
                    # set the test data set with new sample 
                    self.ind_test = np.expand_dims(self.test_data_dic[ind][0][0:3],0)
                    self.test_x = x_input
                    demand = self.test_data_dic[ind][0][3]
                    
                self.test_data_dic[ind] = np.delete(self.test_data_dic[ind],0,axis=0)

                # call dnn forward pass to get r,Q
                start_t = time.time()
                self.dnn_rQ[i,t,:] = np.squeeze(self.get_rq_test(x_input, binary_x=True))
                dnn_time += time.time() - start_t

                self.IL_dnn[i,t] = self.IL_dnn[i,t] - demand

                # L is the lead time_, in scale of a year
                L = int(365*self.l[ind])
                if sum(self.AO_dnn[i,t:]) + self.IL_dnn[i,t] < self.dnn_rQ[i,t,0]:
                    self.AO_dnn[i,t+L] += self.dnn_rQ[i,t,1]
                    fixed_cost_dnn = self.K

                self.dnn_cost[i,t] = max(self.IL_dnn[i,t],0)*self.ch + max(-self.IL_dnn[i,t],0)*self.cp \

                self.IL_dnn[i,t+1] = self.IL_dnn[i,t] + self.AO_dnn[i,t]

        print "dnn \t"
        print self.dis, self.clusters, net_num, "%0.2f" %(sum(np.sum(self.dnn_cost, axis=1)))
        print self.dis, "mean", self.clusters, net_num,  "%0.2f" %(np.mean(self.dnn_cost))
        print self.dis, "std", self.clusters, net_num,  "%0.2f" %(np.std(self.dnn_cost))
        print self.dis, "obs_count", self.clusters, net_num,  "%d" %(obs_count) 
        print self.dis, self.clusters, "cpu times are %0.2f" %(dnn_time)



def restore_and_run(distribution, cluster, nodes, last_run, str_no, max_iter=50000, lr=0.001, lambdaa=0.0005):
    
    # # Cumulative run 
    distributions=['normal', 'beta', 'lognormal', 'exponential', 'uniform']
    clusters=[1,10,100,200]

    dist = [distributions[distribution]][0]

    if dist in ['normal', 'lognormal', 'uniform'] and cluster==100:
        real_cluster = 103
    elif cluster==200:
        real_cluster=203
    else:
        real_cluster = cluster
    print dist, cluster

    # initialize everything
    tf.reset_default_graph()
    config.decay_rate = 0.0005
    config.decay_rate_stair = 0.96     
    config.starter_learning_rate = lr
    config.l2lambda = lambdaa
    config.config.decay_step = 15000

    rq_model=rq(cluster, real_cluster, dist, config)
    rq_model.str_no = str_no
    rq_model.get_data()
    rq_model.get_mu_sigma()
    rq_model.set_rq_settings()
    rq_model.set_dnn_settings()
    rq_model.maxiter = max_iter

    rq_model.power = 0.75
    rq_model.init_momentum = 0.9
    rq_model.nodes = nodes
    rq_model.NoHiLay = len(rq_model.nodes) - 2

    rq_model.create_train_method()
    # initialize everything
    rq_model.restore_dnn()
    rq_model.train()

    if last_run:
        # get knn solution
        rq_model.set_knn_settings(10)
        rq_model.knn_saa()

        # get EIL solution
        rq_model.get_EIL_solution()

        # call dnn forward pass to get r,Q
        rq_model.call_dnn_fp()

        # print result
        rq_model.print_EIL_DNN_KNN_costs()

    dic={}
    dic['lr']=rq_model.lr
    dic['lambda']=rq_model.l2lambda
    dic['node']=rq_model.nodes
    dic['test']=rq_model.test_result[-1]
    dic['str_no']=str_no

    return dic

def run_new_model(distribution, cluster, nodes, last_run, str_no, max_iter=50000, lr=0.001, lambdaa=0.0005):

    # # Cumulative run 
    distributions=['normal', 'beta', 'lognormal', 'exponential', 'uniform']
    clusters=[1,10,100,200]

    dist = [distributions[distribution]][0]

    if dist in ['normal', 'lognormal', 'uniform'] and cluster==100:
        real_cluster = 103
    elif cluster==200:
        real_cluster=203
    else:
        real_cluster = cluster
    print dist, cluster

    tf.reset_default_graph()
    config.decay_rate = 0.0005
    config.decay_rate_stair = 0.96     
    config.starter_learning_rate = lr
    config.power = 0.75
    config.l2lambda = lambdaa
    config.config.decay_step = 15000

    rq_model=rq(cluster, real_cluster, dist, config)
    rq_model.str_no = str_no
    rq_model.get_data()
    rq_model.get_mu_sigma()
    rq_model.set_rq_settings()
    rq_model.set_dnn_settings()
    rq_model.maxiter = max_iter

    rq_model.init_momentum = 0.9
    rq_model.nodes = nodes
    rq_model.NoHiLay = len(rq_model.nodes) - 2

    rq_model.create_train_method()
    rq_model.train()

    if last_run:
        # run knn
        rq_model.set_knn_settings(10)
        rq_model.knn_saa()

        # run EIL     
        rq_model.get_EIL_solution()

        # report the comparisons 
        rq_model.print_EIL_DNN_KNN_costs()

    dic={}
    dic['lr']=rq_model.lr
    dic['lambda']=rq_model.l2lambda
    dic['node']=rq_model.nodes
    dic['test']=rq_model.test_result[-1]
    dic['str_no']=str_no

    return dic


def load_rnn_saved_model(dist, cluster):
    '''
    This function will load and run the saved model for the test dataset, and finally prints the 
    (r,Q) and the corresponding costs
    '''

    if dist in ['normal', 'lognormal', 'uniform'] and cluster==100:
        real_cluster = 103
    elif cluster==200:
        real_cluster=203
    else:
        real_cluster = cluster
    print dist, cluster

    # initialize everything
    tf.reset_default_graph()
    rq_model=rq(cluster, real_cluster, dist, config)
    rq_model.get_data()
    rq_model.get_mu_sigma()
    rq_model.set_rq_settings()
    rq_model.str_no = 0
    rq_model.decay_rate_stair = 0.96
    rq_model.set_dnn_settings()

    # initialize everything
    best_network=None
    if config.best_network == None:
        rq_model.set_network()
        if os.path.exists('rq/best_networks/'):
            cur_dir=os.path.join('rq/best_networks/', rq_model.dis)
        else:
            cur_dir=os.path.join(os.path.join(os.path.abspath('./'),'best_networks/'), rq_model.dis)
        rq_model.model_dir=os.path.join(cur_dir, str(rq_model.clusters))
    else:
        rq_model.nodes = [31, 20 , 2, 2]
        rq_model.model_dir = os.path.join(rq_model.model_dir,str(best_network))

    # create the train method and restore the network
    rq_model.create_train_method()
    rq_model.restore_dnn()

    # call dnn forward pass to get r,Q
    rq_model.call_dnn_fp()
    out = rq_model.test_result


    if 1 == 2:
        # get knn solution
        rq_model.set_knn_settings(10)
        rq_model.knn_saa()

        # get EIL solution
        rq_model.get_EIL_solution()

        # print result
        rq_model.print_EIL_DNN_KNN_costs()

    out = rq_model.test_result
    r = np.unique(out[0][1][:,0])
    q = np.unique(out[0][1][:,1])
    print ""
    print "cost= %0.2f" %(out[0][0])
    print "(r,Q)=", r, q


def call_simulator(config, distribution, cluster):

    distributions=['normal', 'beta', 'lognormal', 'exponential', 'uniform']

    dist = distributions[int(distribution)]

    config.if_print_knn_final_cost = False
    config.if_print_knn_solutions = False
    config.sim_period = 10000


    if cluster == 200:
        cluster_real = 203
    elif dist in ['normal', 'lognormal', 'uniform'] and cluster==100:
        cluster_real = 103
    else:
        cluster_real = cluster

    tf.reset_default_graph()
    rq_model=rq(cluster, cluster_real, dist, config)
    rq_model.get_data()
    rq_model.get_mu_sigma()
    rq_model.set_rq_settings()
    rq_model.run_simulator()


def run_simulator_get_best_network(config, distribution, cluster):
    ''' This function calls '''
    distributions=['normal', 'beta', 'lognormal', 'exponential', 'uniform']

    dist = distributions[int(distribution)]

    config.if_print_knn_final_cost = False
    config.if_print_knn_solutions = False
    config.sim_period = 10000


    if cluster == 200:
        cluster_real = 203
    elif dist in ['normal', 'lognormal', 'uniform'] and cluster==100:
        cluster_real = 103
    else:
        cluster_real = cluster

    tf.reset_default_graph()
    rq_model=rq(cluster, cluster_real, dist, config)

    for i in range(100):
        rq_model.get_data()
        rq_model.get_mu_sigma()
        rq_model.set_rq_settings()        
        rq_model.run_simulator_select_best(i)

def run_dnn_simulator_single_specific_network(config, distribution, cluster, str_num):
    ''' This function calls '''
    distributions=['normal', 'beta', 'lognormal', 'exponential', 'uniform']

    dist = distributions[int(distribution)]

    config.if_print_knn_final_cost = False
    config.if_print_knn_solutions = False
    config.sim_period = 10000


    if cluster == 200:
        cluster_real = 203
    elif dist in ['normal', 'lognormal', 'uniform'] and cluster==100:
        cluster_real = 103
    else:
        cluster_real = cluster

    tf.reset_default_graph()
    rq_model=rq(cluster, cluster_real, dist, config)

    rq_model.get_data()
    rq_model.get_mu_sigma()
    rq_model.set_rq_settings()        
    rq_model.run_simulator_select_best(str_num)

if __name__ == '__main__':
    dist = sys.argv[1] # distribution
    cluster = int(sys.argv[2]) # cluster
    if len(sys.argv) >=4:
        # 1-> it runs all saved network, otherwise it runs the best selected network. 
        run_type = int(sys.argv[3]) 
    else:
        run_type = 0
    if len(sys.argv) >=5:
        # 1-> it runs just a single structure with a given str number, otherwise it runs all structures
        str_num = int(sys.argv[4]) # cluster
    else:
        str_num = 0
    config, unparsed = parser.parse_known_args()

    if str_num != 0:
        run_dnn_simulator_single_specific_network(config, dist, cluster, str_num)
    if run_type == 1:
        run_simulator_get_best_network(config, dist, cluster)
    else:
        # load_run_saved_model(dist, cluster)
        call_simulator(config, dist, cluster)



