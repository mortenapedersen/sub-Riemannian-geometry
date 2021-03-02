from autograd import elementwise_grad as grad
import geomstats.backend as gs
import matplotlib.pyplot as plt

### comments: 

# def metric(q):
#     # the metric which is written on geomstats
#     return gs.array([
#         [1/(1 + q[1]**4) , q[0]**2],
#         [q[0]**2, 1. + 2*q[1]**3]])

def metric(q):
    # 2D Riemannian metric, i.e. a 2x2 matrix
    # q : basepoint in M
    
    return gs.array([
        [1 + q[0] , 0],
        [0, 1 + q[1]]])

def kinetic_energy_Riemannian(x): 
    q, p = x
    cometric = gs.linalg.inv(metric(q))
    return 1/2 * gs.einsum('i,ij,j', p, cometric, p)

def kinetic_energy(x): 
    q, p = x

    frame_matrix = global_r_frame_as_matrix(q) # full Riemannian frame, first 2 rows contains the SR frame
    multiplied = np.einsum('ji,i->j', frame_matrix, p)
    
    
    return 1/2 * (gs.einsum('i,i->',p,[0]['X'])**2 +
                  gs.einsum('i,i->',p,global_r_frame(gs.array([q]))[0]['Y'])**2)

def Hamiltonian(x):
    # x: is on the form np.array([[q_1,q_2],[p_1,p_2]]), where q is interpreted as the M-coordinate
    #    and p is the T^{\star}M-coordinate
    return kinetic_energy(x)

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

def exp_SR(covector, point, n_steps=100):
    # covector : initial cotangent vector 
    # point : initial base-point in M

    x = gs.array([point, covector]) # initial point in T^{\star}M
    return symp_flow(kinetic_energy,n_steps)(x)

def main(init_tangent = gs.array([1.,1.]), init_position = gs.array([0.,0.]), n_steps = 20):
    x = exp_SR(0.05 * init_tangent, init_position, n_steps) #? why the 0.05 factor?
    plt.plot(x[:,0,0], x[:,0,1],'ro-')
    plt.show()
    return x



