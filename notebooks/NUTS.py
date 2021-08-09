import numpy as onp

class HMC:

    def get_u(self, q):
        """Compute the posterior and gradient from (and in) normalized coordinates"""
        # convert from normalized q coordinates to x
        x = self.q2x(q)
        # Get the posterior and gradient
        logP, grad_logP = self.fun(x, *self.fun_args, **self.fun_kwargs)
        # convert the gradient to the transformed coordinates, using the Jacobian.
        # we  don't have to convert logP because the coordinate transformation
        # is linear, so the change to P(x) is just a scaling, so the change to
        # logP is just a constant
        return -logP, -self.L.T @ grad_logP

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

        return q_prime, p_prime

    def find_reasonable_epsilon(self, q):
        #set epsilon and an initial random momentum
        self.epsilon = 1
        p = onp.random.normal(0,1,len(q))

        #update p and q
        q_prime, p_prime = self.leapfrog(q,p)

        #set alpha
        if
        #THIS IS UNFINISHED

    def build_tree(self,q,p,u,v,j,epsilon,q_0,p_0):
        Delta_max = 1000
        if j == 0:
            q_prime, p_prime = leapfrog(q,p,v*epsilon)

            U_prime, gradU_prime = self.get_u(q_prime)
            U_0, gradU_0 = self.get_u(q_0)
            #get nprime
            n_prime = int(u < onp.exp(U_prime - 0.5 * onp.dot(p_prime,p_prime.T)))
            #get sprime
            s_prime = int(u < (Delta_max + onp.exp(U_prime - 0.5 * onp.dot(p_prime,p_prime.T))))

            #gotta be tricksy about returning here
            q_minus, q_plus = q_prime, q_prime
            p_minus, p_plus = p_prime, p_prime
            alpha_prime = onp.min(1, exp(U - 0.5 * onp.dot(p_prime,p_prime.T) - U_0 + 0.5 * onp.dot(p_0,p_0.T)))
            nalpha_prime = 1

        else:
            #recursion
            q_minus,p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, nalpha_prime = build_tree(q,p,u,v,j-1,epsilon,q_0,p_0)

            #step forwards or backwards
            if s_prime == 1:
                if v == -1:
                    q_minus, p_minus, _, _, q_2prime, n_2prime, s_2prime, alpha_2prime, nalpha_2prime = build_tree(q_minus,p_minus,u,v,j-1,epsilon,q_0,p_0)
                else:
                    _, _, q_plus, p_plus, q_2prime, n_2prime, s_2prime, alpha_2prime, nalpha_2prime = build_tree(q_plus,p_plus,u,v,j-1,epsilon,q_0,p_0)

            #accept or reject new step
            if onp.random.uniform() < (float(n_2prime)/(float(n_prime)+float(n_2prime))):
                q_prime = q_2prime

            #update acceptance criteria
            alpha_prime = alpha_prime + alpha_2prime
            nalpha_prime = nalpha_prime + nalpha_2prime

            s_prime = s_2prime * int((q_plus - q_minus)*p_minus >= 0) * int((q_plus - q_minus) * p_plus >= 0)
            n_prime = n_prime + n_2prime

        return q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, nalpha_prime

        def NUTS(self, log_post_and_gradient, q_0, delta, M, Madapt):
            #set some parameters
            epsilon_0 = self.find_reasonable_epsilon(q_0)
            mu = onp.log(10*epsilon_0)
            eps_bar = 1
            H_bar = 0
            gamma = 0.05
            t_0 = 10
            kappa = 0.75
            U, gradU = self.get_u(q_0)

            for m in range(1, M):
                #resample - random kick?
                p_0 = onp.random.normal(0,1,len(q_0))
                u = onp.random.uniform(0,onp.exp(U - 0.5 * onp.dot(p_0, p_0.T)))
