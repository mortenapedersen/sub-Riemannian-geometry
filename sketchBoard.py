import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g = np.array([-1,2,20])
# h = np.array([-2,-3,-2])

n_obs = 50
h = np.ones((n_obs,3))
h = np.transpose(np.multiply(np.transpose(h),np.linspace(0,100,50)))
gh = group_mult(g,h)


##### plotting

### left-translation of a line
fig = plt.figure()
ax  = plt.axes(projection='3d')

data = np.transpose(gh)

min_x = np.amin(np.r_[gh[:,0],h[:,0]]) - 1
max_x = np.amax(np.r_[gh[:,0],h[:,0]]) + 1
min_y = np.amin(np.r_[gh[:,1],h[:,1]]) - 1
max_y = np.amax(np.r_[gh[:,1],h[:,1]]) + 1
min_z = np.amin(np.r_[gh[:,2],h[:,2]]) - 1
max_z = np.amax(np.r_[gh[:,2],h[:,2]]) + 1


ax.axes.set_xlim3d(left=min_x, right=max_x) 
ax.axes.set_ylim3d(bottom=min_y, top=max_y) 
ax.axes.set_zlim3d(bottom=min_z, top=max_z) 

# ax.scatter3D(data[0], data[1],data[2],'-o', cmap='Greens')
ax.plot3D(data[0], data[1],data[2])
ax.plot3D(np.transpose(h)[0],np.transpose(h)[1],np.transpose(h)[2])

plt.show()


### plot the sr frame

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make the grid
x, y, z = np.meshgrid(np.linspace(-2, 2, 10),
                      np.linspace(-2, 2, 10),
                      np.linspace(-2, 2, 10))

# Make the direction data for the arrows
X_u = 1
X_v = 0
X_w = -1/2*y

Y_u = 0
Y_v = 1
Y_w = 1/2*x

ax.quiver(x, y, z, X_u, X_v, X_w, length=0.1, normalize=True, color='red')
ax.quiver(x, y, z, Y_u, Y_v, Y_w, length=0.1, normalize=True, color='blue')

plt.show()

### plots of geodesics ###

fig = plt.figure()
ax  = plt.axes(projection='3d')
points = np.array([[1,1,-1],
                   [1,1,1]])

for i in points:
    point = i
    params = geodesic_to_point(point)
    theta, k, t_final = [params['theta'],params['k'],params['t']]
    grid = np.linspace(0,t_final,100)
    ax.plot3D([geodesic_from_parameters(theta,k,t)[0] for t in grid],
              [geodesic_from_parameters(theta,k,t)[1] for t in grid],
              [geodesic_from_parameters(theta,k,t)[2] for t in grid])

plt.show()



###
val=10
grid = np.linspace(0.4,0.6,200);plt.plot(grid,[np.abs(helper_fun(s,val)) for s in grid])

k0 = 1
theta0 = 0
k1 = -k0
theta1 = 0.64*np.pi
t = 2
grid = np.linspace(0,2,200)
plt.ylim(-2, 2)
plt.xlim(-2, 2)
plt.axes().set_aspect('equal')
plt.grid()
plt.plot([np.cos(theta0)*(np.cos(k0*s)-1)/k0 - np.sin(theta0)*np.sin(k0*s)/k0  for s in grid],[np.sin(theta0)*(np.cos(k0*s)-1)/k0 + np.cos(theta0)*np.sin(k0*s)/k0 for s in grid], color='red')
plt.plot([np.cos(theta1)*(np.cos(k1*s)-1)/k1 - np.sin(theta1)*np.sin(k1*s)/k1  for s in grid],[np.sin(theta1)*(np.cos(k1*s)-1)/k1 + np.cos(theta1)*np.sin(k1*s)/k1 for s in grid], color='blue')
geodesic_from_parameters(theta0,k0,t)
geodesic_from_parameters(theta1,k1,t)

point=np.array([-1,-2,-1])
params = geodesic_to_point(point)
theta, k, t = [params['theta'],params['k'],params['t']]
if (np.round(geodesic_from_parameters(theta,k,t)).astype(int) == point).all():
    print(True)
else:
    print(geodesic_from_parameters(theta,k,t),point)

grid = np.linspace(-1,1,200);plt.plot(grid,[np.abs(s**3) for s in grid])


fig = plt.figure()
ax  = plt.axes(projection='3d')
points = np.array([[1,1,-1],
                   [1,1,1]])

for i in points:
    point = i
    params = geodesic_to_point(point)
    theta, k, t_final = [params['theta'],params['k'],params['t']]
    grid = np.linspace(0,t_final,100)
    ax.plot3D([geodesic_from_parameters(theta,k,t)[0] for t in grid],
              [geodesic_from_parameters(theta,k,t)[1] for t in grid],
              [geodesic_from_parameters(theta,k,t)[2] for t in grid])

plt.show()



#### plot surface of distances

## surface: the xy plane

# generate 2 2d grids for the x & y bounds
n_linspace = 100
linX = np.linspace(-1, 1, n_linspace)
linY = np.linspace(-0.2, 0.2, n_linspace)
x,y = np.meshgrid(linX,linY)
dist = x*0 #initialization of array of distances

for i in range(0,x.shape[1]):
    print(i)
    for j in range(0,x.shape[1]):
        x_0 = x[i,j] 
        y_0 = 0.001
        z_0 = y[i,j]

        point = np.array([x_0,y_0,z_0])

        # print(x_0,y_0,z_0,point)

        try:
            d = SR_distance(point)
        except:
            print(i,j,point,"Didn't work")
            d = -99

        dist[i,j] = d


# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
# dist = dist[:-1, :-1]
dist_min, dist_max = -np.abs(dist).max(), np.abs(dist).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, dist, cmap='RdBu', vmin=dist_min, vmax=dist_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()

# ...

def f(x): return(SR_distance(np.array([x,0,10])))


xx = x[0:30,4]
yy = y[15:30,20]
dd = dist[15:30,20]

plt.plot(xx,dd)
