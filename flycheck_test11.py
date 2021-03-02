import numpy as np
from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.optimize import root_scalar

def group_mult(g,h):
    """
    parameters: g is an array with 3 elements. h is either an array (vector) or a matrix whose rows will be group-multiplied by g (= vectorization).
    output: the 3D array which is the result of group-multiplying g, h.
    """

    if h.shape == (3,): # if h is onedimensional
        return g + h + 1/2*np.array([0,0,g[0]*h[1] - g[1]*h[0]])
    else:
        n_points = h.shape[0]

        output = np.zeros((n_points,3)) + 9999 # initialize output
    
        for i in range(n_points):
            output[i,:] = g + h[i,:] + 1/2*np.array([0,0,g[0]*h[i,1] - g[1]*h[i,0]])
       
        return output


def tangent_translation_map(point):
    """
    Parameters: [3-array] point to evaluate the left-translation in. The left-translation is then lifted to its differential linear map (matrix).
    Returns: [3x3 array] the differential.
    """

    return(np.array([[1,0,0],
                     [0,1,0],
                     [-point[1]/2,point[0]/2,1]]))


def global_r_frame(point):
    """
    A global orthonormal frame for the Riemannian mannifold. The first two elements
    of the frame is the canonical sub-Riemannian frame, generating the distribution.

    parameters: [3-array], or an nx3 array. The point(s) in which to evaluate the frame.
    returns: dict with 3 arrays, 'X','Y','Z'. Each array is the corresponding frame vector (see Donne).
o    """

    # frame = dict()

    # frame['X'] = np.array([1,0,-1/2*point[1]])
    # frame['Y'] = np.array([0,1,1/2*point[0]])
    # frame['Z'] = np.array([0,0,1])


    frames = [{'X': np.array([1,0,-1/2*p[1]]),
     'Y': np.array([0,1,1/2*p[0]]),
     'Z': np.array([0,0,1])} for p in point]  # check: a list of dictionaries is probably very inefficient (?)
                          
    return frames


def global_r_frame_as_matrix(point):
    """
    A global orthonormal frame for the Riemannian mannifold. The first two elements
    of the frame is the canonical sub-Riemannian frame, generating the distribution.

    parameters: [3-array], or an nx3 array. The point(s) in which to evaluate the frame.
    returns: dict with 3 arrays, 'X','Y','Z'. Each array is the corresponding frame vector (see Donne).
o    """

    # frame = dict()

    # frame['X'] = np.array([1,0,-1/2*point[1]])
    # frame['Y'] = np.array([0,1,1/2*point[0]])
    # frame['Z'] = np.array([0,0,1])


    frame_matrix = np.array([[1,0,-1/2*p[1]],
                       [0,1,1/2*p[0]],
                             [0,0,1]])  # check: a list of dictionaries is probably very inefficient (?)
                          
    return frame_matrix


def inner_product_matrix(point):
    """
    the inner product matrix corresponding to the basis (1.2.6) in Donne, written in standard
    euclidean coordinates.
    """
    x, y = point[0:2]

    return 1/4*np.array([[y**2+4,-x*y,2*y],
                  [-x*y,x**2 + 4, -2*x],
                  [2*y,-2*x,4]])


def helper_fun(s,val):
    """
    helper-function to find inverse of H. (see paper by ...)
    """
    return 2*np.pi/(1-np.cos(2*np.pi*s)) * (s - np.sin(2*np.pi*s)/(2*np.pi)) - val

def helper_fun_inv(val):
    """ 
    inverse function of H.
    """

    candidates = np.array([-1+ 10**(-10),1-10**(-10)])
    counter = 0
    max_iter = 20
    while np.amax(candidates) - np.amin(candidates) > 10**(-8) and counter < max_iter:
        counter += 1 
        grid = np.linspace(candidates[0],candidates[-1],5000)
        fct = np.array([helper_fun(s,val) for s in grid]) #plt.plot(grid,fct)
        # candidates_prev = candidates
        # candidates = grid[np.abs(fct) < 10**(-power)]
        candidates = grid[np.argmin(np.abs(fct))-1:np.argmin(np.abs(fct))+1]

    # output = root(helper_fun, x0=0.1, args=(val)).x[0]
    # output = least_squares(helper_fun, (0.1), bounds = ([0],[1]), args={'val':val})
    # output = root_scalar(helper_fun,args=(val),method='bisect', bracket=(-1,1), x0=0.5).root
        
    # return candidates_prev[np.argmin(np.array([np.abs(helper_fun(s,val)) for s in candidates_prev]))]
    return candidates[0]

def SR_distance(point):
    """
    find the sub-Riemannian distance between the origin and point.
    """

    x,y,z = point

    norm = np.sqrt(x**2+y**2)
    H_inv = helper_fun_inv(4*z*norm**(-2))  # nb: *4 indsat hacket for maaske at faa det til at passe med hajlasz' parametrisering

    ### rgt hack... indsat minus paa output, hvis z er neg
    if z >= 0:
        return 2*np.pi*H_inv*norm/(np.sqrt(2*(1-np.cos(2*np.pi*H_inv))))
    # return z*np.sin(np.pi*H_inv)*norm**(-1) + norm*np.cos(np.pi*H_inv)
    else:
        return -2*np.pi*H_inv*norm/(np.sqrt(2*(1-np.cos(2*np.pi*H_inv))))



def k_root(point,t):
    """
    find initial guess os roots by simple function-evaluation on a grid
    """

    upper_lim = 2*np.pi/t # cf. p21 (below) Donne
    n_evals = 5000

    grid = np.linspace(10**(-30),upper_lim,n_evals)
    fct = np.array([np.abs(k_fun(k,point,t)) for k in grid])
    # visualization:  plt.plot(grid,fct)

    candidates = grid[fct<np.amin(fct)*100]
    
    root_min, root_max = [np.amin(candidates),np.amax(candidates)]

    return {'min': root_min, 'max': root_max}

    
def geodesic_to_point(point):
    """
    parameters:
      'point'; array (endpoint of geodesic)
      'resolution'; float (distance between points along the curve)
    return: parameters for the geodesic
    """

    point_orig = point

    point = np.array([point[0],point[1],np.abs(point[2])])  # first, the problem is solved for positive z 

    t = SR_distance(point)

    k_roots = k_root(point,t)
    k_min_0,k_max_0 = [k_roots['min'],k_roots['max']]
    k_range = [k_min_0 - k_min_0/10, k_max_0 + (2*np.pi/t - k_max_0)/10]
        
    def equations(vars):
        theta, k = vars
        eq1 = np.cos(theta)*(np.cos(k*t) - 1)/k - np.sin(theta)*(np.sin(k*t)/k) - point[0]
        eq2 = np.sin(theta)*(np.cos(k*t) - 1)/k + np.cos(theta)*(np.sin(k*t)/k) - point[1]
        # eq3 = (k*t-np.sin(k*t))/(2*k**2) - point[2]
        return [eq1, eq2]

    theta,k = least_squares(equations, (np.pi, k_min_0), bounds = ((0, k_range[0]), (2*np.pi, k_range[1]))).x


    if point_orig[2] < 0:
        k = -k

        endpoint_x_y_temp = [(np.cos(k*t)-1)/k, np.sin(k*t)/k]
        endpoint_x_y_final = np.array([point[0],point[1]])

        endpoint_matrix = np.array([[endpoint_x_y_temp[0],-endpoint_x_y_temp[1]],
                                    [endpoint_x_y_temp[1],endpoint_x_y_temp[0]]])

        if np.linalg.cond(endpoint_matrix) < 1/sys.float_info.epsilon:
            endpoint_matrix_inv = np.linalg.inv(endpoint_matrix)

            cos_theta, sin_theta = np.matmul(endpoint_matrix_inv, endpoint_x_y_final)

            theta = np.arccos(cos_theta)
            theta_0 = np.arcsin(sin_theta)

            print("theta,theta_0 equal? : ",theta,theta_0)
            
        else: print("theta-matrix not invertible, solve it brute force, tbd")
                  
    
    return {'theta':theta,'k':k,'t':t}

point = np.array([1,1,2])

def equations(vars):
    theta, k, t = vars
    eq1 = np.cos(theta)*(np.cos(k*t) - 1)/k - np.sin(theta)*(np.sin(k*t)/k) - point[0]
    eq2 = np.sin(theta)*(np.cos(k*t) - 1)/k + np.cos(theta)*(np.sin(k*t)/k) - point[1]
    eq3 = (k*t-np.sin(k*t))/(2*k**2) - point[2]
    return [eq1, eq2, eq3]

theta, k, t = fsolve(equations, (1, 0.5,0.1))

def geodesic_from_parameters(theta,k,t):
    return np.array([np.cos(theta)*(np.cos(k*t) - 1)/k - np.sin(theta)*(np.sin(k*t)/k),
                     np.sin(theta)*(np.cos(k*t) - 1)/k + np.cos(theta)*(np.sin(k*t)/k),
                     (k*t-np.sin(k*t))/(2*k**2)]) 

def k_fun(k,point,t):
    return (k*t-np.sin(k*t))/(2*k**2) - point[2]

def theta_fun(theta,point,k,t):
    return np.cos(theta)*(np.cos(k*t) - 1)/k - np.sin(theta)*(np.sin(k*t)/k) - point[0]



# t = SR_distance(point)


# k_roots = k_root(point,t)
# k_min_0,k_max_0 = [k_roots['min'],k_roots['max']]

# k_min = fsolve(k_fun, (k_min_0), args = (point,t))[0] # plt.plot(np.linspace(0,3,100),[np.abs((k*t-np.sin(k*t))/(2*k**2) - point[2]) for k in np.linspace(0,3,100)])
# k_max = fsolve(k_fun, (k_max_0), args = (point,t))[0]

# grid = np.linspace(0,2*np.pi,100)
# plt.plot(grid,[theta_fun(theta,point,k_min,t) for theta in grid])

# theta = fsolve(equations2, (2.3))[0]

# geodesic_from_parameters(theta,k,t)

# ## - check at laengden er t (altsaa at den er laengde-parametriseret) - hvis ikke, kan denne repr maaske slet ikke bruges?
# ##    ... det burde dog holde: jeg saetter jo bare parametre ind i formlen som gaelder for det vi VED er tids-parametriserede geodaeter. Det eneste sanity-check er om endpointet er det onskede.
# ## - formlen crasher for diverse vaerdier ... check hvorfor: det er vist for neg z, hvor theta vaerdierne af en eller anden grund er forskellige.



# geodesic_from_parameters(theta,k,t)







# timepoints = np.linspace(0,4,100)
# k = -1
# plt.plot([(np.cos(k*t)-1)/k for t in timepoints],[(np.sin(k*t))/k for t in timepoints])
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# plt.axes().set_aspect('equal')
