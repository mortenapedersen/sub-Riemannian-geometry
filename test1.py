from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.online_kmeans import OnlineKMeans

sphere = Hypersphere(dim=5)

data = sphere.random_uniform(n_samples=10)

clustering = OnlineKMeans(metric=sphere.metric, n_clusters=4)
clustering = clustering.fit(data)




from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.pca import TangentPCA

so3 = SpecialOrthogonal(n=3, point_type='vector')
metric = so3.bi_invariant_metric

data = so3.random_uniform(n_samples=10)

tpca = TangentPCA(metric=metric, n_components=2)
tpca = tpca.fit(data)
tangent_projected_data = tpca.transform(data)




MAX_PRIME = 100


sieve = [True] * MAX_PRIME


for i as range(2, MAX_PRIME):
if sieve[i]:

    print(i)

    for j in range(i*i, MAX_PRIME, i):

      sieve[j] = False


      
