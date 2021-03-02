import numpy as np

x = np.array([[1.1,1.1],[2.2,2.1]])
# x = np.array([[1.1],[2.2]])


def Ham(x):
    fun = kinetic_energy
    # fun = np.sin
    val = fun(x)
    return val

symp_grad(Ham)(x)

init_tangent = vector = covector = gs.array([1.,1.,1.]); init_position = point = gs.array([0.,0.,0]); n_steps = 2
main(init_tangent = gs.array([1,0,0]), init_position = gs.array([0.,0.,0.]), n_steps = 2)
