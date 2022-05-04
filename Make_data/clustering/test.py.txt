from sklearn.cluster import KMeans
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv("probes.csv", index_col = 0)

cost =[]
for i in range(1, 39):
    KM = KMeans(n_clusters = i)
    KM.fit(df)
     
    # calculates squared error
    # for the clustered points
    cost.append(KM.inertia_)    
 
# plot the cost against K values
plt.figure(figsize=(16,9))
plt.plot(range(1, 39), cost, color ='g', linewidth ='3')
plt.xlabel("Value of K")
plt.ylabel("Squared Error (Cost)")
plt.savefig("tmp.png") # clear the plot