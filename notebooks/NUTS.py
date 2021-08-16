import numpy as onp

class HMC:
    def __init__(self, log_post_and_gradient, covariance_estimate, limits, args=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if args is None:
            args = []
        self.fun = log_post_and_gradient
        self.fun_args = args
        self.fun_kwargs = kwargs

        # The mid-point.
        # We use this to transform the boundary planes
        self.limits = onp.array(limits)
        self.x0 = onp.array([0.5*(lim[0] + lim[1]) for lim in limits])
        self.ndim = len(self.x0)

        # We use this to transform between the original space and the
        # transformed one.  We use this often enough that it's worth
        # doing the full transform here.
        # We transform the q coordinates instead of the p momenta,
        # since otherwise the velocities and the momenta can point
        # in different directions, and this messes up the reflection
        # in a way I couldn't figure out.
        self.L = onp.linalg.cholesky(covariance_estimate)
        self.Linv = onp.linalg.inv(self.L)

        # We trace the accepted samples
        self.trace_logP = []
        self.trace = []
        self.trace_accept = []

        # As well as tracing the accepted samples it's also nice
        # to trace the integration paths between those samples.
        # Improvements to the algorithm make use of those.
        self.paths = []
        self.paths_logP = []

        self.ncall = 0
        self.ncall_list = []
    
    def q2x(self, q):
        """Transform a vector in the space with diagonal mass to a parameter vector"""
        return self.L @ q + self.x0

    def x2q(self, x):
        """Transform a parameter vector to a vector in the space with diagonal mass"""
        return self.Linv @ (x - self.x0)

    def stop_when(self, q_minus, q_plus, p_minus, p_plus):
        dq = q_plus - q_minus
        criterion_1 = onp.dot(dq,p_minus.T) >= 0
        criterion_2 = onp.dot(dq,p_plus.T) >= 0
        return criterion_1 & criterion_2

    def get_u(self, q):
        """Compute the posterior and gradient from (and in) normalized coordinates"""
        # convert from normalized q coordinates to x
        x = self.q2x(q)
        #x=q
        # Get the posterior and gradient
        logP, grad_logP = self.fun(x, *self.fun_args, **self.fun_kwargs)
        self.ncall += 1
        # convert the gradient to the transformed coordinates, using the Jacobian.
        # we  don't have to convert logP because the coordinate transformation
        # is linear, so the change to P(x) is just a scaling, so the change to
        # logP is just a constant
        return logP, self.L.T @ grad_logP
        #return logP, grad_logP

    def leapfrog(self, q, p, epsilon):
        #get gradient
        U, gradU = self.get_u(q)
        #first half-update
        p_prime = p + 0.5*epsilon*gradU
        #update position with half-updated mom
        q_prime = q + epsilon * p_prime
        #get new gradient
        U_new, gradU_new = self.get_u(q_prime)
        #second half-update
        p_prime = p_prime + 0.5*epsilon*gradU_new

        return q_prime, p_prime, U_new, gradU_new

    def find_reasonable_epsilon(self, q):
        #set epsilon and an initial random momentum
        epsilon = 1.
        p_0 = onp.random.normal(0.,1.,len(q))
        #get U
        U, gradU = self.get_u(q)

        #update p and q
        _, p_prime, U_prime, gradU_prime = self.leapfrog(q,p_0, epsilon)

        #set alpha
        condition = U_prime-U-0.5*(onp.dot(p_prime,p_prime)-onp.dot(p_0,p_0))
        if condition > onp.log(0.5):
            alpha = 1.
        else:
            alpha = -1.

        while alpha * condition > -alpha * onp.log(2.):
            #update epsilon
            epsilon = epsilon * (2. ** alpha)
            print(epsilon)
            #simulate step
            _, p_prime, U_prime, _ = self.leapfrog(q,p_0,epsilon)
            #update condition
            condition = U_prime-U-0.5*(onp.dot(p_prime,p_prime)-onp.dot(p_0,p_0))

        print("Reasonable epsilon is "+str(epsilon))

        return epsilon


    def build_tree(self,q,p,u,v,j,epsilon,q_0,p_0):
        Delta_max = 1000
        if j == 0:
            q_prime, p_prime, U_prime, gradU_prime = self.leapfrog(q,p,v*epsilon)

            condition = U_prime - 0.5 * onp.dot(p_prime,p_prime.T)

            U_0, gradU_0 = self.get_u(q_0)
            condition_0 = U_0 - 0.5 * onp.dot(p_0,p_0.T)

            #get nprime
            n_prime = int(u < condition)
            #get sprime
            s_prime = int(u < (Delta_max + condition))

            #return stuff
            alpha_prime = min(1., onp.exp(condition - condition_0))
            nalpha_prime = 1
            return q_prime, p_prime, q_prime, p_prime, q_prime, n_prime, s_prime, alpha_prime, 1

        else:
            #recursion
            q_minus,p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, nalpha_prime = self.build_tree(q,p,u,v,j-1,epsilon,q_0,p_0)

            #step forwards or backwards
            if s_prime == 1:
                if v == -1:
                    q_minus, p_minus, _, _, q_2prime, n_2prime, s_2prime, alpha_2prime, nalpha_2prime = self.build_tree(q_minus,p_minus,u,v,j-1,epsilon,q_0,p_0)
                else:
                    _, _, q_plus, p_plus, q_2prime, n_2prime, s_2prime, alpha_2prime, nalpha_2prime = self.build_tree(q_plus,p_plus,u,v,j-1,epsilon,q_0,p_0)

                #accept or reject new step
                if onp.random.uniform() < max(float(int(n_prime) + int(n_2prime)), 1.):
                    q_prime = q_2prime

                #update acceptance criteria
                alpha_prime = alpha_prime + alpha_2prime
                nalpha_prime = nalpha_prime + nalpha_2prime

                s_prime = int(s_2prime and self.stop_when(q_minus, q_plus, p_minus, p_plus))
                n_prime = n_prime + n_2prime

        return q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, nalpha_prime

    def NUTS(self, q_0, delta, M, M_adapt):
        #set some parameters
        q_0 = self.x2q(q_0)
        epsilon = self.find_reasonable_epsilon(q_0)
        mu = onp.log(10*epsilon)
        eps_bar = 1
        H_bar = 0
        gamma = 0.05
        t_0 = 10
        kappa = 0.75
        U, gradU = self.get_u(q_0)

        #make arrays to later hold stuff
        samples = onp.empty((M+M_adapt,len(q_0)))

        samples[0,:] = q_0

        for m in range(1, M + M_adapt):
            if m %((M+M_adapt)/20) == 0:
                print("Step %s of %s" %(m, M+M_adapt))
                #print("epsilon is " +str(epsilon))

            if m == M_adapt+2:
                print("final epsilon is " +str(epsilon))
            #resample - random kick?
            p_0 = onp.random.normal(0,1,len(q_0))
            condition = U - 0.5 * onp.dot(p_0, p_0.T)
            #u = condition - onp.random.exponential(1,size=1)
            u = onp.log(onp.random.uniform(0,onp.exp(condition), size=1))

            #set m to m-1 step
            samples[m, :] = samples[m - 1, :]

            #initialise
            q_minus, q_plus = samples[m-1,:], samples[m-1,:]
            p_minus, p_plus = p_0, p_0
            j, n, s = 0, 1, 1

            while s == 1:
                #choose direction
                v = onp.random.choice([-1,1])

                if v == -1:
                    q_minus, p_minus, _, _, q_prime, n_prime, s_prime, alpha, nalpha = self.build_tree(q_minus, p_minus, u, v, j, epsilon, samples[m-1,:], p_0)
                else:
                    _, _,q_plus, p_plus, q_prime, n_prime, s_prime, alpha, nalpha = self.build_tree(q_plus, p_plus, u, v, j, epsilon, samples[m-1,:], p_0)

                #apply MH acceptance
                if s_prime == 1:
                    if onp.random.uniform() < min(1, float(n_prime) / float(n)):
                        samples[m, :] = q_prime
                        #print("Sample accepted j = " + str(j))
                        #HAVE TO CONVERT Q BACK TO X; DUH
                        self.paths.append(self.q2x(q_prime))
                        

                n = n + n_prime
                s = s_prime and self.stop_when(q_minus, q_plus, p_minus, p_plus)
                j += 1

            if m <= M_adapt:
                fraction = 1/(m+t_0)
                H_bar = (1. - fraction) * H_bar + fraction * (delta - alpha / float(nalpha))
                epsilon = onp.exp(mu - onp.sqrt(m) / gamma * H_bar)
                power = m ** -kappa
                eps_bar = onp.exp(power * onp.log(epsilon) + (1-power) * onp.log(eps_bar))
                if m%500000000 == 0:
                    print(epsilon)
            else:
                epsilon = eps_bar

            self.trace.append(self.q2x(samples[m, :]))
            self.ncall_list.append(self.ncall)
            

 
        #self.trace = samples[M_adapt:,:]
        #print(samples)
