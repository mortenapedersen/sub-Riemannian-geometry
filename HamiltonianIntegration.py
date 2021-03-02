from autograd import elementwise_grad as grad
import geomstats.backend as gs
import matplotlib.pyplot as plt

### comments:
# ...
### questions:
# - where the connection..? R-geodesics are not determined only by the metric, are they..?
#   maybe if the L-C connection is implicitly used?

# def metric(q):
#     # the metric which is written on geomstats
#     return gs.array([
#         [1/(1 + q[1]**4) , q[0]**2],
#         [q[0]**2, 1. + 2*q[1]**3]])

def metric(q):
    # Riemannian metric, i.e. a nxn matrix
    # q : basepoint in M, dim(M) = np.size(q)

    # 3D example
    return gs.array([
        [1 + q[0] , 0, 0],
        [0, 1 + q[1],0],
        [0, 0, 1 + q[2]]])


def kinetic_energy(x):  # example of a Hamiltonian
    # x: is on the form np.array([[q_1,q_2],[p_1,p_2]]), where q is interpreted as the M-coordinate
    #    and p is the T^{\star}M-coordinate
    q, p = x
    cometric = gs.linalg.inv(metric(q))
    return 1/2 * gs.einsum('i,ij,j', p, cometric, p)


def symp_grad(H):
    # i.e. the Hamiltonian vector field of H
    def vector(x): 
        H_q, H_p = grad(H)(x) # grad = elementwise_grad is vectorized gradient-calculation
        return gs.array([H_p, - H_q])
    return vector

def symp_euler(H):
    #the update-step in 'symplectic euler integration': https://en.wikipedia.org/wiki/Semi-implicit_Euler_method
    #output: a function which takes input x_n (position), and outputs the next position x_{n+1}
    
    def step(x):
        q, p = x
        dq, _ = symp_grad(H)(x)
        y = gs.array([q + dq, p]) 
        _, dp = symp_grad(H)(y) # that dp is calculated from the updated q-position is the difference from standard euler integration.
        return gs.array([q + dq, p + dp])

    return step


def iterate(func, n_steps):
    # helper-function (not specifically related to (symplectic) geometry)

    def flow(x):
        xs = [x] 
        for i in range(n_steps):
            xs.append(func(xs[i]))
        return gs.array(xs)

    return flow

def symp_flow(H, n_steps=20): 
    return iterate(symp_euler(H), n_steps)

def exp(vector, point, n_steps=100):
    # vector : initial tangent vector (mapped to a covector)
    # point : initial base-point in M
    
    momentum = gs.einsum('ij,j', metric(point), vector) # this maps 'vector' to the corresponding covector (under the metric isomorphism)
    x = gs.array([point, momentum]) # initial point in T^{\star}M
    return symp_flow(kinetic_energy,n_steps)(x)

def main(init_tangent = gs.array([1.,1.,1]), init_position = gs.array([0.,0.,0.]), n_steps = 20):
    x = exp(0.05 * init_tangent, init_position, n_steps)
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    ax.plot3D(x[:,0,0], x[:,0,1],x[:,0,2],'ro-)
    # plt.plot(x[:,0,0], x[:,0,1],x[:,0,2],'ro-')
    plt.show()
    return x




###### problematic output for the following inputs
## using the original metric in the geomstats code - NB: the mentioned behavior is probably due to the fractions in the metric
# main(np.array([1.,1.]), np.array([0.,0.]), 20) # wrong initial direction ([-1,1] instead of [1,1])
# main(np.array([0.2,0.2]), np.array([0.,0.]), 200) # weird behavior after some amount of time-steps
